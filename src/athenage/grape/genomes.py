import random
import copy

class Genome:
    """
    A GE genome.
    """
    def __init__(self, n_rules:int, codon_size:int):
        self.codon_size = codon_size
        self.init_codons(n_rules)

    def init_codons(self, n_rules:int) -> None:
        self.codons = []
        self.next_read=0
    
    def add_codon(self,val:int, rule_used:int) -> None:
        self.codons.append(val)

    def finalize(self) -> None:
        pass
        
    def size(self) -> int:
        return len(self.codons)

    def add_tail(self, n_codons:int) -> None:
        for i in range(n_codons):
            self.codons.append(random.randint(0,self.codon_size))
    
    def random_fill(self,genome_length:int) -> None:
        """ Fill genome with random codons """
        for i in range(genome_length):
            self.codons.append(random.randint(0, self.codon_size))

    def consumed(self, rule_used:int) -> bool:
        "Returns True if all codons have been used"
        return True if self.next_read >= len(self.codons) else False

    def reset_map_index(self) -> None:
        """Reset the genome so that get_next_codon starts with first codon"""
        self.next_read = 0

    def get_next_codon(self, rule_used:int) -> int:
        """ Get next codon from genome for mapping

        Args:
            rule_used: index of rule used in grammar
        
        Returns: 
            codon: codon value 
        """
        codon = self.codons[self.next_read]
        self.next_read += 1

        return codon

    def used_codons(self) -> int:
        return self.next_read
    
    def total_codons(self)-> int:
        return len(self.codons)
    
    def set_codon(self, codon_idx:int, value:int, invalid:bool) -> None:
        self.codons[codon_idx]=value

    def effective_cross_loc(self) -> list:
        """Return a list that contains the position for crossover"""
        return [random.randint(1,self.next_read)]

    def all_cross_loc(self) -> list:
        """Return location for a cross over entire genome"""
        return [random.randint(1,len(self.codons))]

    def crossover_onepoint(self, genome2:Genome, pos1:list, pos2:list) -> tuple[Genome,Genome]:
        """ cross over genomes and return new ones 
        Args:
            genome2:second geome to cross with this one
            pos1: codon position for cross on in the genome
            pos2: codon position for cross on second genome

        Returns:
            new_genome1: Genome
            new_genome2: Genome   
        """

        new_genome1 = copy.deepcopy(self)
        new_genome2 = copy.deepcopy(genome2)

        new_genome1.codons = self.codons[0:pos1[0]] + genome2.codons[pos2[0]:]
        new_genome2.codons = genome2.codons[0:pos2[0]] + self.codons[pos1[0]:]

        return new_genome1, new_genome2

class LeapGenome(Genome):
    """
    A GE genome.
    """
    def __init__(self, n_rules:int, codon_size:int):
        super().__init__(n_rules, codon_size)

    def init_codons(self, n_rules:int) -> None:
        super().init_codons(n_rules)
        self.frame_size = n_rules
        self.codons.extend(self.new_frame())
        self.last_frame = 0
        
    def new_frame(self)  -> list:
        """ Return new frame initialized to be empty """
        return [-1 for i in range(self.frame_size)]
    
    def add_codon(self,val:int, rule_used:int)  -> None:
        codon_idx = self.last_frame * self.frame_size + rule_used
        if self.codons[codon_idx] != -1:
            # fill unused codons with random values
            self.codons[-self.frame_size:] = [random.randint(0,self.codon_size) if x == -1 else x for x in self.codons[-self.frame_size:]]
            self.codons.extend(self.new_frame())
            self.last_frame += 1
            codon_idx += self.frame_size

        self.codons[codon_idx] = val
    
    def finalize(self)  -> None:
        """Convert any unset codons to hold random values"""
        self.codons[-self.frame_size:] = [random.randint(0,self.codon_size) if x == -1 else x for x in self.codons[-self.frame_size:]]
    
    def add_tail(self, n_codons:int)  -> None:
        """Convert n_codons to complete frames"""
        n_codons = n_codons + self.frame_size - (n_codons % self.frame_size)
        for i in range(n_codons):
            self.codons.append(random.randint(0,self.codon_size))

    def random_fill(self,genome_length:int) -> None:
        """ Fill genome with frames containing random codons"""
        genome_length = genome_length + (genome_length % self.frame_size)
        for i in range(genome_length):
            self.codons.append(random.randint(0, self.codon_size))

    def consumed(self,rule_used:int) -> bool:
        "Returns True if all codons have been used"
        if ((self.last_frame * self.frame_size + self.frame_size > len(self.codons)) or
        (self.consumed_codons[rule_used] and ((self.last_frame+1) * self.frame_size + self.frame_size > len(self.codons)))):
            return True
        else:
            return False

    def reset_map_index(self) -> None:
        """Reset the genome so that get_next_codon starts with first frame"""
        self.last_frame = 0
        self.consumed_codons = [False for i in range(self.frame_size)]

    def get_next_codon(self, rule_used:int) -> int:
        """ Get next codon from genome for mapping

        Args:
            rule_used: index of rule used in grammar
        
        Returns: 
            codon: codon value or False if no next codon
        """
        if(self.consumed_codons[rule_used]):
            self.last_frame += 1
            self.consumed_codons = [False for i in range(self.frame_size)]

        self.consumed_codons[rule_used] = True
        codon_idx = self.last_frame * self.frame_size + rule_used
        return self.codons[codon_idx]

    def used_codons(self) -> int:
        return self.last_frame * self.frame_size
    
    def effective_cross_loc(self) -> list:
        """Return a list that contains the frame for crossover"""
        return [random.randint(1,self.last_frame)]

    def all_cross_loc(self) -> list:
        """Return location for a cross over entire genome"""
        return [random.randint(1,len(self.codons)//self.frame_size-1)]

    def crossover_onepoint(self, genome2: LeapGenome, pos1:list, pos2:list) -> tuple[LeapGenome,LeapGenome]:
        """ cross over genomes and return new ones 
        Args:
            genome2: LeapGenome to cross with
            pos1: contains frame to cross for this genome
            pso2: contains frame to cross at second genome

        Returns;
            new_genome1: LeapGenome
            new_genome2: LeapGenome    
        """
        new_genome1 = copy.deepcopy(self)
        new_genome2 = copy.deepcopy(genome2)

        pt1 = pos1[0] * self.frame_size
        pt2 = pos2[0] * self.frame_size

        new_genome1.codons = self.codons[0:pt1] + genome2.codons[pt2:]
        new_genome2.codons = genome2.codons[0:pt2] + self.codons[pt1:]

        return new_genome1, new_genome2


class MCGEGenome(Genome):
    """
    A GE genome.
    """
    def __init__(self, n_rules, codon_size):
        super().__init__(n_rules, codon_size)
        
    def init_codons(self, n_rules) -> None:
        self.codons = [ [] for i in range(n_rules)]
        self.consumed_codons = [ 0 for i in range(n_rules)]

    def add_codon(self,val:int, rule_used:int) -> None:
        self.codons[rule_used].append(val)

    
    def size(self) -> int:
        """Returns average size of a chromosome"""
        total = 0
        for chr in self.codons:
            total += len(chr)
        return total // len(self.codons) + 1

    def add_tail(self, n_codons:int) -> None:
        """ Add tail to each chrom"""
        for chr in self.codons:
            for i in range(n_codons):
                chr.append(random.randint(0,self.codon_size)) 

    def random_fill(self,genome_length:int) -> None:
        """ Fill genome with chromosomes that equal genome length"""
        chrom_length = genome_length // len(self.codons) + 1
        for i in range(len(self.codons)):
            for j in range(chrom_length):
                self.codons[i].append(random.randint(0, self.codon_size))

    def consumed(self,rule_used:int) -> bool:
        "Returns True if all codons have been used"
        return True if self.consumed_codons[rule_used] >= len(self.codons[rule_used]) else False

    def reset_map_index(self) -> None:
        """Reset the genome so that get_next_codon starts with first frame"""
        self.consumed_codons = [ 0 for i in range(len(self.codons))]

    def get_next_codon(self, rule_used:int) -> int:
        """ Get next codon from genome for mapping

        Args:
            rule_used: index of rule used in grammar
        
        Returns: 
            codon: codon value or False if no next codon
        """
        codon = self.codons[rule_used][self.consumed_codons[rule_used]]
        self.consumed_codons[rule_used] += 1
        return codon

    def used_codons(self) -> int:
        return sum(self.consumed_codons)
    
    def total_codons(self)-> int:
        total = 0
        for chr in self.codons:
            total += len(chr)
        return total
    
    def set_codon(self, codon_idx:int, value:int, invalid:bool) -> None:
        """ Sets specified codon. Calculates chromosome and position for codon to match index passed
        
        Args:
            codon_idx: overall codon position to alter
            value: new codon value
            invalid: when True use all codons to calculate position
        
        Returns:
            None
        """
        if invalid:
            for chrnum in range(len(self.codons)):
                if codon_idx >= len(self.codons[chrnum]):
                    codon_idx -= (len(self.codons[chrnum])+1)
                else:
                    self.codons[chrnum][codon_idx] = value
        else:
            for chrnum in range(len(self.consumed_codons)):
                if codon_idx >= self.consumed_codons[chrnum]:
                    codon_idx -= (self.consumed_codons[chrnum]+1)
                else:
                    self.codons[chrnum][codon_idx] = value
                    break

    def effective_cross_loc(self) -> list:
        """Return a list that contains the frame for crossover"""
        positions = []
        for chr_idx in range(len(self.codons)):
            positions.append(random.randint(0,self.consumed_codons[chr_idx]))
        return positions

    def all_cross_loc(self) -> list:
        """Return location for a cross over entire genome"""
        positions = []
        for chr_idx in range(self.codons):
            positions.append(random.randint(1,len(self.codons[chr_idx])))
        return positions

    def crossover_onepoint(self, genome2: MCGEGenome, pos1:list, pos2:list) -> tuple[MCGEGenome,MCGEGenome]:
        """ cross over genomes and return new ones 
        Args:
            genome2: MCGEGenome to cross with
            pos1: contains [chrom,position] to cross for this genome
            pso2: contains [chrom,position]  to cross at second genome

        Returns;
            new_genome1: MCGEGenome created after crossover
            new_genome2: MCGEGenome created after crossover
        """
        new_genome1 = copy.deepcopy(self)
        new_genome2 = copy.deepcopy(genome2)

        #select chromosome for crossover
        chr_idx = random.randint(0,len(self.codons)-1)

        new_genome1.codons[chr_idx] = self.codons[chr_idx][0:pos1[chr_idx]] + genome2.codons[chr_idx][pos2[chr_idx]:]
        new_genome2.codons[chr_idx]  = genome2.codons[chr_idx][0:pos2[chr_idx]] + self.codons[chr_idx][pos1[chr_idx]:]
        return new_genome1, new_genome2