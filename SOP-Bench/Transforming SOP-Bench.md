## Transforming SOP-Bench

**SOP-Bench** is a dataset designed to evaluate AI agents on realistic, multi-step workflows expressed as Standard Operating Procedures SOPs. The transformation of this dataset is similar to what we observe in WorfBench, as both rely on a list of actions and an action graph representation. 

However, SOP-Bench introduces additional control structure: an OR node. This OR represents alternative execution paths, meaning that action A or action B can be performed. To handle this in our transformation, we adopt a sequential decision strategy. We first execute action A, then introduce a decision node that checks whether A succeeded. If it succeeds, the workflow continues; otherwise, the alternative path (action B) is executed. This process can be extended to multiple alternatives.

Table illustrating a simple example of this mapping logic is provided in the paper.

![Mapping example between SOP workflow and its corresponding SysML v2 code](Mapping%20example%20.png)