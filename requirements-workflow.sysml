package RequirementManagementWorkflow {

    private import ScalarValues::*;

    action def GenerateRequirementsProcess {
        
        first start;
        
        then action userInput {
            description = "Get a system description from the user.";
            out systemDescription : String;
        }

        then loop action constructRequirementList  {
            in systemDescription : String;
            inout requirements : String [0..*];

            action decompose {
                description = "A new requirement is generated based on the system description.";
                in systemDescription : String;
                out newRequirement : String;
            }

            then fork; then checkCorrectness; then checkNonRedundancy; then checkSatisfiability;
            
            action checkCorrectness {
                description = "Checks whether the new Requirement is correct based on the system description.";
                in newRequirement : String;
                in systemDescription : String;
                out isCorrect : Boolean;
            }
            then joinNode;

            action checkNonRedundancy {
                description = "Checks that the new requirement is not redundant based on the current list of accepted requirements";
                in newRequirement : String;
                in systemDescription : String;
                in requirements : String [0..*];
                out isNonRedundant : Boolean;
            }
            then joinNode;

            action checkSatisfiability {
                description = "Checks whether the new requirement is satisfiable.";
                in newRequirement : String;
                in systemDescription : String;
                in requirements : String [0..*];
                out isSatisfiable : Boolean;
            }
            then joinNode;

            join joinNode;

            then action checkConsensus {
                in isCorrect : Boolean;
                in isNonRedundant : Boolean;
                in isSatisfiable : Boolean;
                out allValid = isCorrect and isNonRedundant and isSatisfiable;
            }

            flow decompose.newRequirement to updateRequirements.newRequirement;

            then action updateRequirements {
                description = "Updates the list of accepted requirements and checks whether the list of accepted requirements is complete.";
                in requirements : String [0..*];
                in newRequirement : String;
                in systemDescription : String;
                out isComplete : Boolean;
            }
        } until constructRequirementList.updateRequirements.isComplete;

        then action outputToUser {
            description = "Returns the list of completed requirements to the user.";
            inout requirements : String [0..*];
        }

        then done;
    }
}
