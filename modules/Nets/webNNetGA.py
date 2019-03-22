"""
webNNet method definitions for everything GA related

@author: ljknoll
"""


import torch
import numpy as np
import SkyNEt.modules.Evolution as Evolution

def trainGA(self, 
            train_data,
            target_data,
            cf,
            loss_fn = None,
            verbose = False):
    """ Train web with Genetic Algorithm """
    
    train_data, target_data = self.check_cuda(train_data, target_data)
    
    self.check_graph()
    
    # prepare config object with information of web, # of genes, partitions, genomes, etc
    self.prepare_config_obj(cf, loss_fn)
    # initialize genepool
    genepool = Evolution.GenePool(cf)
    # stores which indices of self.parameters to change during training
    self.set_dict_indices_from_pool(genepool.pool[0])
    
    # np arrays to save genePools, outputs and fitness
    geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
    outputArray = np.zeros((cf.generations, cf.genomes, train_data.shape[0], self.nr_output_vertices))
    fitnessArray = np.zeros((cf.generations, cf.genomes))
    
    # Temporary arrays, overwritten each generation
    fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
    outputAvg = torch.zeros(cf.fitnessavg, train_data.shape[0], self.nr_output_vertices, device=self.cuda)
    outputTemp = torch.zeros(cf.genomes, train_data.shape[0], self.nr_output_vertices, device=self.cuda)

    for i in range(cf.generations):
        for j in range(cf.genomes):
            for avgIndex in range(cf.fitnessavg):
                # update parameters of each network
                self.set_parameters_from_pool(genepool.pool[j])
                self.forward(train_data)
                outputAvg[avgIndex] = self.get_output()
                # use negative loss as fitness for genepool.NextGen()
                fitnessTemp[j, avgIndex] = -cf.Fitness(outputAvg[avgIndex], target_data).item()
                
            outputTemp[j] = outputAvg[np.argmin(fitnessTemp[j])]
        
        genepool.fitness = fitnessTemp.min(1)  # Save best fitness

        # Save generation data
        geneArray[i, :, :] = genepool.pool
        outputArray[i, :, :] = outputTemp
        fitnessArray[i, :] = genepool.fitness

        if verbose:
            print("Generation nr. " + str(i + 1) + " completed")
            print("Best fitness: " + str(-max(genepool.fitness)))
        
        genepool.NextGen()

    return geneArray, outputArray, fitnessArray


def prepare_config_obj(self, cf, loss_fn):
    """ prepares config object for GA use with the web class """
    # total number of genes: 5 control voltages for each vertex, one less for each arc
    cf.genes = len(self.graph)*5 - len(self.arcs)
    # number of genomes in each of the 5 partitions
    cf.partition = [5, cf.genes, 5, 5, cf.genes]
    # total number of genomes
    cf.genomes = sum(cf.partition)
    
    # set fitness functino of cf to default loss
    # loss function must return the error, not the fitness!
    if loss_fn is None:
        cf.Fitness = self.loss_fn
    else:
        cf.Fitness = loss_fn

def set_dict_indices_from_pool(self, pool):
    """ For each registered parameter in this web object, 
    store the indices of the parameters which are used.
    
    Example:
        when there is an arc from 'A' to 'B' at gate 5, 
        the GA does not need to update this control voltage as it is overwritten by the arc data.
        Therefore, these parameters are not included in the pool.
        The indices for A will then be [2,3,4,6] and gate 5 will not be passed to evolution
    """
    parameters = self.get_parameters()
    indices = {}
    # loop through parameters of network
    for par_name, par_params in parameters.items():
        indices[par_name] = []
        # loop through control voltages of vertex 'par_name'
        for j in range(len(par_params)):
            # check if current control voltage is in use by an arc, thus no value is needed
            # TODO: replace +2 by actual number of datainputs of network
            if (par_name, j+2) not in self.arcs.keys():
                indices[par_name].append(j)
    
    self.indices = indices

def set_parameters_from_pool(self, pool):
    """ Uses the indices set by set_dict_indices_from_pool() to 
    update parameters with values from pool """
    pool_iter = iter(pool)
    with torch.no_grad(): # do not track gradients
        for par_name, indices in self.indices.items():
            # replace parameter par_name values with values from pool
            replacement = [next(pool_iter) for _ in range(len(indices))]
            getattr(self, par_name)[indices] = torch.tensor(replacement, dtype=torch.float32, device=self.cuda)
            
            
            
            
def noveltyGA(self,
              train_data, 
              cf,
              initial_archive_size,
              novelty_threshold,
              threshold_update_interval=None,
              k_nearest_neighbors=5,
              genomes_added_threshold = 50,
              novelty_threshold_dec=0.95,
              novelty_threshold_inc=1.2,
              normalize=False,
              verbose=False):
    """
    Implementation of part of the novelty search of the CHARC framework, see 
    'A Substrate-Independent Framework to Characterise Reservoir Computers'
    
    arguments:
    train_data              data used as input
    cf                      configuration object for GA
    initial_archive_size    archive is initialized with random control voltages
    novelty_threshold       fitness threshold determining if the current cv is added to archive
    threshold_update_interval (optional, int) after how many generations to do a threshold update
    k_nearest_neighbors     (optional, int) how many nearest neighbors to look at when calculating novelty
    genomes_added_threshold (optional, int) if nr_genomes added is more than this threshold, novelty_threshold is increased by 20%
    novelty_threshold_dec   (optional, float) factor which novelty_threshold is decreased
    novelty_threshold_inc   (optional, float) factor which novelty_threshold is increased
    normalize               (optional, bool) whether to use and return normalized archive_output
    verbose                 (optional, bool) whether to print information during search
    
    returns:
    geneArray               all genes visited
    outputArray             all outputs visited
    fitnessArray            fitness values of outputArray
    archive                 archive of control voltages
    archive_output          outputs of archive
    novelty_threshold       novelty threshold after searching (changes during)
    """
    
    if threshold_update_interval == None:
        threshold_update_interval = np.ceil(cf.generations/10)

    train_data = self.check_cuda(train_data)

    self.check_graph()

    # prepare config object with information of web, # of genes, partitions, genomes, etc
    self.prepare_config_obj(cf, None)

    # archive with actual genes
    archive = np.zeros((initial_archive_size+cf.generations*cf.genomes, cf.genes))
    # archive with behaviour of genes, index (maximum size of archive, train_data length, nr of output vertices)
    archive_output = torch.zeros(initial_archive_size+cf.generations*cf.genomes, 
                                 train_data.shape[0], 
                                 self.nr_output_vertices)

    def sparseness(y_pred, archive_output, index):
        # find k nearest neighbors and return mean distance
        distances, _ = torch.topk((archive_output[:index]-y_pred)**2, min(k_nearest_neighbors, index), dim=0, largest=False)
        # first mean over all elements in archive, second mean over all values along one output dimension
        return torch.mean(torch.mean(distances, dim = 0)**0.5, dim=0)

    # initialize genepool
    genepool = Evolution.GenePool(cf)
    # stores which indices of self.parameters to change during training
    self.set_dict_indices_from_pool(genepool.pool[0])

    # np arrays to save genePools, outputs and fitness
    geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
    outputArray = np.zeros((cf.generations, cf.genomes, train_data.shape[0], self.nr_output_vertices))
    fitnessArray = np.zeros((cf.generations, cf.genomes))

    # Temporary arrays, overwritten each generation
    fitnessTemp = np.zeros((cf.genomes, cf.fitnessavg))
    outputAvg = torch.zeros(cf.fitnessavg, train_data.shape[0], self.nr_output_vertices, device=self.cuda)
    outputTemp = torch.zeros(cf.genomes, train_data.shape[0], self.nr_output_vertices, device=self.cuda)

    with torch.no_grad():
        for i in range(initial_archive_size):
            self.reset_parameters('rand')
            archive[i] = np.random.rand(cf.genes)
            self.set_parameters_from_pool(archive[i])
            self.forward(train_data)
            temp = self.get_output(False, False)
            if normalize:
                temp = (temp-torch.mean(temp, dim=0))/torch.std(temp, dim=0)
            archive_output[i] = temp
        current_size = initial_archive_size
        nr_genomes_added = 0
        for i in range(cf.generations):
            for j in range(cf.genomes):
                for avgIndex in range(cf.fitnessavg):
                    # update parameters of each network
                    self.set_parameters_from_pool(genepool.pool[j])
                    self.forward(train_data)
                    outputAvg[avgIndex] = self.get_output()
                    fitnessTemp[j, avgIndex] = sparseness(outputAvg[avgIndex], archive_output, current_size).item()

                outputTemp[j] = outputAvg[np.argmax(fitnessTemp[j])]

            genepool.fitness = fitnessTemp.max(1)  # Save best fitness of averages

            archive_indices = np.where(genepool.fitness > novelty_threshold)
            nr_genomes_to_add = len(archive_indices[0])
            if nr_genomes_to_add != 0:
                # add novel genomes
                archive[current_size:current_size+nr_genomes_to_add] = genepool.pool[archive_indices]
                temp = outputTemp[archive_indices]
                if normalize:
                    temp = (temp-torch.mean(temp, dim=0))/torch.std(temp, dim=0)
                archive_output[current_size:current_size+nr_genomes_to_add] = temp
                current_size += nr_genomes_to_add
                nr_genomes_added += nr_genomes_to_add

            # update threshold
            if i%threshold_update_interval == threshold_update_interval-1:
                if nr_genomes_added == 0:
                    novelty_threshold *= novelty_threshold_dec
                elif nr_genomes_added > genomes_added_threshold:
                    novelty_threshold *= novelty_threshold_inc
                nr_genomes_added = 0

            # Save generation data
            geneArray[i, :, :] = genepool.pool
            outputArray[i, :, :] = outputTemp
            fitnessArray[i, :] = genepool.fitness

            if verbose:
                print("Generation nr. " + str(i + 1) + " completed")
                print("Best fitness: " + str(max(genepool.fitness)))

            genepool.NextGen()

    return geneArray, outputArray, fitnessArray, archive[:current_size], archive_output[:current_size], novelty_threshold
