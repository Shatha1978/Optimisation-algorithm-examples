import GeneticOperator
import Individual as IND


class GaussianMutationOperator(GeneticOperator.GeneticOperator):

    # Contructor
    def __init__(self, aProbability, aMutationVariance):
        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "Gaussian mutation operator";

        # Set the mutation variance
        self.mutation_variance = aMutationVariance;

    def getMutationVariance(self):
        return self.mutation_variance;

    def setMutationVariance(self, aMutationVariance):
        self.mutation_variance = aMutationVariance;

    def apply(self, anEA):

        self.use_count += 1;
        
        # Select the parents from the population
        parent_index = anEA.selection_operator.select(anEA.individual_set)

        # Copy the parent into a child
        child = anEA.individual_set[parent_index];

        # Mutate the child and return it
        return self.mutate(child);

    def mutate(self, anIndividual):
        anIndividual.gaussianMutation(aMutationRate);
        return anIndividual;
