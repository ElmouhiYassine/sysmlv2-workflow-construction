# Transforming SAP-SAM

**SAP-SAM** is a large dataset containing around 1.02M diagrams, including UML and BPMN models. In this work, we focus on transforming BPMN diagrams (about 631.4k in total) into SysML v2 code, under a set of defined conditions.

## Language Filtering

We first restrict the dataset to BPMN diagrams written in English. However, a significant portion of the dataset does not have an assigned language. To address this, we use the `langdetect` NLP package to automatically detect the language.

For each diagram, we aggregate the textual descriptions of its tasks and classify it as English if the detected probability exceeds 0.8. After this filtering step, approximately 342k BPMN diagrams remain.

## Element Filtering

We then restrict the dataset to diagrams that only use a subset of BPMN elements that can be mapped to SysML v2. Applying this constraint reduces the dataset to approximately 56.6k diagrams, which constitutes the final set for transformation.

## Transformation Design Decisions

BPMN diagrams are primarily designed for human interpretation, and many elements such as conditions are expressed in natural language. In contrast, SysML v2 requires formal boolean conditions.

To bridge this gap, we introduce dedicated actions to represent these conditions. Instead of directly translating condition expressions into formal logic, we encapsulate them into actions that return a boolean value. For example, a BPMN condition such as "check if panel is not empty" is transformed into an action (e.g., `check_panel`) with the same description, which returns a boolean value used in decision nodes.

### The mapping between BPMN elements and their corresponding SysMLv2 representations is summarized in the following table:
![Mapping between BPMN elements and their corresponding SysML v2 representations
](mapping_bpmn_to_sysml.png)

## Generated Dataset Size

Using this transformation pipeline, we generated approximately **56.2k SysML v2 workflow models** derived from SAP-SAM.