# Comprehensive Analysis Report: Agency Effects on Memory

> **Note**: This is a markdown version of the comprehensive statistical analysis report. For the original formatted HTML version with full styling, see [`comprehensive_analysis_report.html`](comprehensive_analysis_report.html).

---

# Comprehensive Analysis Report: Agency Effects on Memory


*Generated: 2025-11-29 12:59:36*


*This report contains the manuscript main results section with full statistics inserted from analyses run1 through run13.*



---


## RUN 1: Agency did not improve recall performance



For each participant, events were binned according to whether they were remembered or forgotten.
Independent raters compared each sentence of recall to the story path read by the participant;
if any part of a given event was mentioned in any recall sentence, it was counted as remembered.

**Statistical Results:**
```
Adventure: F(2,113) = 1.433, p = 0.243
Romance: F(2,123) = 0.671, p = 0.513
```


There were no significant differences in recall performance across conditions in either story
(**Adventure: F(2,113)=1.43, p=0.243**;
**Romance: F(2,123)=0.67, p=0.513**;
*Supplementary Figure S3*). The Romance story additionally recorded
individual reading speed and showed no difference across the three agency conditions (ps> .3;
see **Supplement S3** for details). See **Supplement S4**
for details about memory for choice and non-choice events; See **Supplement S5**
for details about recognition memory performance; See **Supplement S6**
for details about memory for denied and granted choice events.






---


## RUN 2: Individual variability in recalled events



**Agency magnified individual variability in recall and choice.** The Romance story,
by design, had half of its events shared across all participants ("shared story sections"),
regardless of condition (*Figure 1A*). While participants made many
choices during these shared story sections, unbeknownst to them, all choice options led to the
same subsequent events. This allowed us to examine inter-participant variability in terms of
memory (which events were recalled) and choice behavior (which options were selected) when all
events were perfectly matched across participants, i.e., all participants read these events,
and the events were composed of identical text.



**Individual variability in recalled events.** A recall score (0 = Forgotten, 1 = Recalled;
see Methods for details) for each of the 64 events in the shared story sections was extracted for each
participant, composing a vector of recall performance (*Figure 2A*).
To assess the memory similarity across participants, we computed the inter-participant correlation (ISC),
i.e., the Pearson correlation between each pair of participants' recall performance vectors, i.e.,
"Recall ISC".

**64 Shared Events - One-sample t-tests (Raw values):**
```
FREE: mean r = 0.136, t(152) = 11.746, p = 0.000000
YOKE: mean r = 0.226, t(1377) = 58.288, p = 0.000000
PASV: mean r = 0.249, t(1484) = 70.871, p = 0.000000
```

**64 Shared Events - ANOVA (Raw values):**
```
F(2,3013) = 48.127, p = 0.000000
**Post-hoc t-tests:**
Free vs Yoked: t(1529.000) = -7.285, p = 0.000000
Free vs Passive: t(1636.000) = -9.731, p = 0.000000
Yoked vs Passive: t(2861.000) = -4.463, p = 0.000008
```

**49 Non-Choice Events - One-sample t-tests (Raw values):**
```
FREE: mean r = 0.157, t(152) = 13.212, p = 0.000000
YOKE: mean r = 0.219, t(1377) = 51.429, p = 0.000000
PASV: mean r = 0.229, t(1484) = 59.028, p = 0.000000
```

**49 Non-Choice Events - ANOVA (Raw values):**
```
F(2,3013) = 15.445, p = 0.000000
**Post-hoc t-tests:**
Free vs Yoked: t(1529.000) = -4.632, p = 0.000004
Free vs Passive: t(1636.000) = -5.679, p = 0.000000
Yoked vs Passive: t(2861.000) = -1.739, p = 0.082
```

**64 Shared Events - One-sample t-tests (Fisher z-transformed):**
```
FREE: mean z = 0.140, t(152) = 11.672, p = 0.000000
YOKE: mean z = 0.235, t(1377) = 55.827, p = 0.000000
PASV: mean z = 0.260, t(1484) = 67.258, p = 0.000000
```

**64 Shared Events - ANOVA (Fisher z-transformed):**
```
F(2,3013) = 45.955, p = 0.000000
**Post-hoc t-tests:**
Free vs Yoked: t(1529.000) = -7.190, p = 0.000000
Free vs Passive: t(1636.000) = -9.496, p = 0.000000
Yoked vs Passive: t(2861.000) = -4.322, p = 0.000016
```

**49 Non-Choice Events - One-sample t-tests (Fisher z-transformed):**
```
FREE: mean z = 0.161, t(152) = 13.077, p = 0.000000
YOKE: mean z = 0.229, t(1377) = 49.029, p = 0.000000
PASV: mean z = 0.239, t(1484) = 55.878, p = 0.000000
```

**49 Non-Choice Events - ANOVA (Fisher z-transformed):**
```
F(2,3013) = 14.975, p = 0.000000
**Post-hoc t-tests:**
Free vs Yoked: t(1529.000) = -4.636, p = 0.000004
Free vs Passive: t(1636.000) = -5.600, p = 0.000000
Yoked vs Passive: t(2861.000) = -1.609, p = 0.108
```


While events differed in their overall memorability, Recall ISC was significantly above zero
in all three conditions (**Romance: Free: mean r = 0.136, p




---


## RUN 12: Permutation test for Recall ISC



In the Yoked and Passive conditions, multiple participants followed the story-path corresponding
to each of the 18 unique Free participant story-paths. To ensure that the above-reported higher
inter-participant memory similarity in the Yoked and Passive conditions was not due to participants
sharing the same story-path, we conducted non-parametric tests of Recall ISC
(*Figure 2B-C*). We randomly sampled one Yoked and one Passive participant
from each of the 18 story-paths to form a sample of 18 Yoked and 18 Passive participants; thus, no pairs
within these samples read the same story-path. This process was repeated 10,000 times to generate
distributions of Recall ISC for both the Yoked and Passive conditions.

**================================================================================**
```
PERMUTATION TEST FOR RECALL ISC
================================================================================
Method: Non-parametric permutation test
- Randomly sampled 1 Yoked and 1 Passive participant from each of 18 story-paths
- Repeated 10,000 times to generate distributions
- Compared Free condition's Recall ISC to these distributions
Random seed: 42 (for reproducibility)
================================================================================
ANALYSIS: 64 Shared Events
================================================================================
Free condition mean ISC: 0.1364
Permutation distributions:
Yoked: Mean = 0.2183, Std = 0.0201, N = 10000
Passive: Mean = 0.2530, Std = 0.0180, N = 10000
Statistical tests:
Free vs Yoked: p = 0.000000 (proportion of Yoked samples below Free mean)
Result: Free ISC significantly lower than Yoked (p
```


The analysis confirmed that the Free condition's Recall ISC was significantly lower than that of
the Passive (**p




---


## RUN 3: Individual variability in choices made



The option that was selected (1 or 2) at each of the 15 choice-points in the shared story sections
was extracted for each Free and Yoked participant (Passive participants made no choices), composing a
vector of choice selections. To assess the choice similarity across participants, we computed the
inter-participant (Pearson) correlation between each pair of participants' choice selection vectors,
i.e., "Choice ISC".

**Choice ISC Results (Raw values):**
```
FREE: mean r = 0.208, t(152) = 9.548, p = 0.000000
YOKE: mean r = 0.255, t(1377) = 36.193, p = 0.000000
Free vs Yoked: t = -2.105, p = 0.035
```

**Choice ISC Results (Fisher z-transformed):**
```
FREE: mean z = 0.233, t(152) = 9.133, p = 0.000000
YOKE: mean z = 0.289, t(1377) = 31.175, p = 0.000000
Free vs Yoked: t = -1.934, p = 0.053
```


Within-group Choice ISC was significantly above zero in both conditions (ps




---


## RUN 13: Permutation test with matching Choice ISC



This increased Choice ISC within the Free condition did not drive their increased Recall ISC.
Another permutation test showed that Free participants still had significantly reduced within-group
Recall ISC compared to their Yoked counterparts with matching Choice ISC
(**p




---


## RUN 4: Divergence from the group



Each Free participant's memory divergence score was calculated as one minus the Pearson correlation
between their recall performance vector and the group averaged recall performance vector. In other words,
the more different their memory performance vector was from the group average, the higher their memory
divergence score. Similarly, we calculated each Free participant's choice divergence score as one minus
the Pearson correlation between their choice selection vector and the group averaged choice selection
vector.

**Memory Divergence vs Choice Divergence Correlation:**
```
free18: r(16) = 0.405, p = 0.095
free100: r(98) = 0.296, p = 0.0028
```


Memory divergence and choice divergence were correlated with each other
(**Romance: r(16) = .405, p = .095**;
**r(98) = .296, p = .003**). In other words, the more idiosyncratic
their memory for shared events, the more idiosyncratic their choices.



Overall, these results support the idea that agency magnified individual variability in, i.e.,
personalized, both memory and choice behaviors. This effect was observed while all events were
held constant across conditions.






---


## RUN 5: Event recall was predicted by semantic and causal centrality



Narrative networks were computed for each unique story path following the methods of Lee & Chen.
For semantic narrative network analysis, each event was converted into an embedding vector using the
Universal Sentence Encoder (USE). Semantic centrality, a measure of how strongly interconnected a
given event was with other events in the narrative via shared meaning, was calculated for each event
by averaging its embedding cosine similarity with all other events in the story path. The effect of
semantic centrality (semantic influence) on memory was computed as the Pearson correlation between
semantic centrality and recall (an event-by-event vector of remembered=1, forgotten=0) for each participant.

**Semantic and Causal Centrality - One-sample t-tests (Raw values):**
```
Adventure FREE Semantic: mean r = 0.223, t(21) = 5.582, p = 0.000015
Adventure FREE Causal: mean r = 0.187, t(21) = 4.604, p = 0.000153
Adventure YOKE Semantic: mean r = 0.323, t(44) = 16.071, p = 0.000000
Adventure YOKE Causal: mean r = 0.159, t(44) = 5.453, p = 0.000002
Adventure PASV Semantic: mean r = 0.308, t(48) = 12.681, p = 0.000000
Adventure PASV Causal: mean r = 0.126, t(48) = 4.922, p = 0.000011
Romance FREE Semantic: mean r = 0.146, t(17) = 6.109, p = 0.000012
Romance FREE Causal: mean r = 0.213, t(17) = 8.613, p = 0.000000
Romance YOKE Semantic: mean r = 0.221, t(52) = 19.021, p = 0.000000
Romance YOKE Causal: mean r = 0.236, t(52) = 14.398, p = 0.000000
Romance PASV Semantic: mean r = 0.265, t(54) = 20.003, p = 0.000000
Romance PASV Causal: mean r = 0.244, t(54) = 23.988, p = 0.000000
```

**Semantic and Causal Centrality - One-sample t-tests (Fisher z-transformed):**
```
Adventure FREE Semantic: mean z = 0.234, t(21) = 5.483, p = 0.000019
Adventure FREE Causal: mean z = 0.194, t(21) = 4.554, p = 0.000173
Adventure YOKE Semantic: mean z = 0.342, t(44) = 15.031, p = 0.000000
Adventure YOKE Causal: mean z = 0.166, t(44) = 5.381, p = 0.000003
Adventure PASV Semantic: mean z = 0.328, t(48) = 11.968, p = 0.000000
Adventure PASV Causal: mean z = 0.132, t(48) = 4.854, p = 0.000013
Romance FREE Semantic: mean z = 0.149, t(17) = 6.062, p = 0.000013
Romance FREE Causal: mean z = 0.220, t(17) = 8.169, p = 0.000000
Romance YOKE Semantic: mean z = 0.227, t(52) = 18.431, p = 0.000000
Romance YOKE Causal: mean z = 0.244, t(52) = 13.866, p = 0.000000
Romance PASV Semantic: mean z = 0.275, t(54) = 19.035, p = 0.000000
Romance PASV Causal: mean z = 0.251, t(54) = 23.059, p = 0.000000
```


For both stories, semantic centrality significantly predicted recall, i.e., a significant semantic
influence on memory was observed, in all three conditions (ps




---


## RUN 6: Agency reduced the influence of semantic but not causal centrality on recall



We next compared the strength of semantic and causal influences on memory across the three conditions
(Free, Yoked, Passive). Importantly, because Yoked and Passive participants read the story paths generated
by Free participants, event content was matched across conditions; only the degree of perceived agency varied.

**Semantic Centrality - One-way ANOVA (Raw values):**
```
Adventure: F(2.000,113) = 3.036, p = 0.052
**Post-hoc t-tests:**
Adventure Free Vs Yoke: t(65) = -2.503, p = 0.015
Adventure Free Vs Pasv: t(69) = -1.889, p = 0.063
Romance: F(2.000,123) = 11.455, p = 0.000027
**Post-hoc t-tests:**
Romance Free Vs Yoke: t(69) = -3.090, p = 0.0029
Romance Free Vs Pasv: t(71) = -4.429, p = 0.000034
Romance Yoke Vs Pasv: t(106) = -2.493, p = 0.014
```

**Semantic Centrality - One-way ANOVA (Fisher z-transformed):**
```
Adventure: F(2.000,113) = 2.870, p = 0.061
**Post-hoc t-tests:**
Adventure Free Vs Yoke: t(65) = -2.441, p = 0.017
Adventure Free Vs Pasv: t(69) = -1.888, p = 0.063
Romance: F(2.000,123) = 11.343, p = 0.000030
**Post-hoc t-tests:**
Romance Free Vs Yoke: t(69) = -3.066, p = 0.0031
Romance Free Vs Pasv: t(71) = -4.371, p = 0.000042
Romance Yoke Vs Pasv: t(106) = -2.531, p = 0.013
```

**Causal Centrality - One-way ANOVA (Raw values):**
```
Adventure: F(2.000,113) = 0.864, p = 0.424
Romance: F(2.000,123) = 0.638, p = 0.530
```

**Causal Centrality - One-way ANOVA (Fisher z-transformed):**
```
Adventure: F(2.000,113) = 0.809, p = 0.448
Romance: F(2.000,123) = 0.562, p = 0.572
```

**Repeated Measures ANOVA - Interaction (Raw values):**
```
Adventure: F(2.000,113) = 3.272, p = 0.042
Romance: F(2.000,123) = 2.990, p = 0.054
```

**Repeated Measures ANOVA - Interaction (Fisher z-transformed):**
```
Adventure: F(2.000,113) = 3.175, p = 0.046
Romance: F(2.000,123) = 3.079, p = 0.050
```


For both stories, we observed significant differences across conditions, wherein Free had lower
semantic influence on memory compared to Yoked and Passive
(**Adventure: F(2,113) = 3.04, p = 0.052**;
**Romance: F(2,123) = 11.46, p




---


## RUN 7: Agency introduces temporal dependencies in memory



We examined whether recall performance for a given event could be predicted by whether its temporally
neighboring events at encoding were recalled, which we term the "neighbor encoding effect". First, for
each participant and for each event, we calculated the average of the recall scores for the immediately
previous and next events at encoding (the neighbors); for the first and last event, there were neighbors
on only one side, and thus these entries consisted merely of recall performance for the next and previous
event, respectively. This procedure generated a vector of neighbor recall performance for each participant.
We then calculated the neighbor encoding effect as the correlation between the neighbor recall performance
vector and the original recall performance vector for each participant (*Figure 4A*).

**Neighbor Encoding Effect - One-sample t-tests (Raw values):**
```
Adventure:
FREE: mean r = 0.282, t(21.000) = 6.103, p = 0.000005
YOKE: mean r = 0.235, t(44.000) = 6.193, p = 0.000000
PASV: mean r = 0.239, t(48.000) = 8.714, p = 0.000000
Romance:
FREE: mean r = 0.352, t(17.000) = 7.589, p = 0.000001
YOKE: mean r = 0.186, t(52.000) = 7.409, p = 0.000000
PASV: mean r = 0.126, t(54.000) = 6.426, p = 0.000000
```

**Neighbor Encoding Effect - One-sample t-tests (Fisher z-transformed):**
```
Adventure:
FREE: mean z = 0.310, t(21.000) = 5.625, p = 0.000014
YOKE: mean z = 0.259, t(44.000) = 6.030, p = 0.000000
PASV: mean z = 0.254, t(48.000) = 8.421, p = 0.000000
Romance:
FREE: mean z = 0.383, t(17.000) = 7.216, p = 0.000001
YOKE: mean z = 0.197, t(52.000) = 7.095, p = 0.000000
PASV: mean z = 0.130, t(54.000) = 6.306, p = 0.000000
```

**Neighbor Encoding Effect - One-way ANOVA (Raw values):**
```
Adventure: F(2.000,113.000) = 0.368, p = 0.693
Romance: F(2.000,123.000) = 12.073, p = 0.000016
**Post-hoc t-tests:**
Romance free vs yoke: t(69.000) = 3.255, p = 0.0018
Romance free vs pasv: t(71.000) = 5.238, p = 0.000002
```

**Neighbor Encoding Effect - One-way ANOVA (Fisher z-transformed):**
```
Adventure: F(2.000,113.000) = 0.400, p = 0.671
Romance: F(2.000,123.000) = 12.556, p = 0.000011
**Post-hoc t-tests:**
Romance free vs yoke: t(69.000) = 3.266, p = 0.0017
Romance free vs pasv: t(71.000) = 5.388, p = 0.000001
```


The neighbor encoding effect was positive in all three conditions for both stories
(Adventure and Romance, ps




---


## RUN 8: Temporal violation rate



The neighbor encoding effect is distinct from the "temporal contiguity effect", which describes
the phenomenon that recalling one item from a randomized list tends to trigger the recall of items
which were experienced nearby in time; the neighbor encoding effect does not incorporate any information
about the temporal order of recall. To examine temporal order effects during recall in our data, we
calculated the temporal violation rate as the frequency with which each participant recalled events out
of order.

**Temporal Violation Rate - One-way ANOVA:**
```
Adventure: F(2,113) = 2.564, p = 0.081
Romance: F(2,123) = 0.096, p = 0.908
```


For each participant, recall was divided into segments (brief sentences). We counted the number of
times that a recall segment referred to an event that occurred earlier in the story than the events
referred to by the previous recall segment; this was then divided by the participant's total number of
recall segments. There was no significant difference in temporal violation rate across conditions in
either story (**Adventure: F(2,113) = 2.56, p = .081**;
**Romance: F(2,123) = 0.10, p = .913**). In general, temporal order was
remarkably well-preserved, with low temporal violation rates in all conditions
(*Figure 4C-D*).






---


## RUN 9: Greater memory divergence was associated with weaker semantic influence



We next examined how memory divergence scores (*Figure 2*) were related
to a) the impact of semantic centrality on recall (*Figure 3*) and b) the
neighbor encoding effect (*Figure 4*). Each Free participant's semantic
influence score was obtained by calculating the Pearson correlation between semantic centrality and memory
performance (same as shown in *Figure 3*).



Memory divergence was negatively correlated with semantic influence scores in Free participants. In other
words, the more a participant's recall deviated from other participants in the group, the weaker the effect
of semantic centrality on their recall.

**Memory Divergence vs Semantic Influence:**
```
N=18: r(16) = -0.519, p = 0.027
N=100: r(98) = -0.460, p = 0.000001
```


This was true when including only the 18 Free participants who had yoked counterparts
(**Romance: r(16) = -.519, p = 0.027**) as well as when using the full sample
(**Romance: r(98) = -.460, p




---


## RUN 10: Neighbor encoding effect correlations



The neighbor encoding effect was also positively associated with memory divergence in Free participants
(**Romance: r(16) = 0.496, p = .036**;
**r(98)= 0.304, p = .002**;
*Supplementary Figure S10B*). In other words, the more a participant's
recall deviated from other participants in the group, the more that participant's neighboring events tended
to have the same recall status (remembered or forgotten).

**Neighbor Encoding Effect Correlations:**
```
Adventure (N=22): Neighbor Encoding Effect vs Semantic Influence: r(20) = -0.348, p = 0.113
Romance (N=18): Neighbor Encoding Effect vs Memory Divergence: r(16) = 0.496, p = 0.036
Romance (N=18): Neighbor Encoding Effect vs Semantic Influence: r(16) = -0.324, p = 0.189
Romance (N=100): Neighbor Encoding Effect vs Memory Divergence: r(98) = 0.304, p = 0.0021
Romance (N=100): Neighbor Encoding Effect vs Semantic Influence: r(98) = -0.347, p = 0.000402
```

**Multiple Linear Regression (Romance, N=100):**
```
Neighbor Encoding Effect: p = 0.087
Semantic Influence: p = 0.000047
```


The neighbor encoding effect was negatively correlated with semantic influence scores in Free participants,
across both stories (**Adventure: r(22) = -.348, p = .113**.
**Romance: r(16) = -.324, p = .189**;
**r(98)= -.347, p




---


## RUN 11: Consequences of having your choices denied



In the results reported above, Yoked subjects generally exhibited similar memory performance to the Passive
subjects—similar idiosyncrasy in event recall, in semantic and causal centrality effect on memory, and in neighbor
encoding effects, with their group mean falling in between that of the Free and Passive condition. Given that the
Yoked condition had a varied amount of choice granted/denied (see *Supplementary Figure S6-1*),
these results aligned with the design of 'partial agency' for the Yoked condition in a gradient of agency.



Nonetheless, having one's agency denied can have unique memory effects at local choice events: one's memory for
the denied choice events is selectively reduced compared to its choice-granted counterparts in the Free condition

**Agency Denial Effect - Two-sample t-test:**
```
Adventure: p = 0.015
Romance: p = 0.017
```

(**Adventure: p = 0.015**;
**Romance: p = 0.017**, two-sample t-test); individual differences contribute
to a variation of tendency to selective recall or forget the denied choice events (see
**Supplement S6** for details).



The percentage of choices granted in the Yoked participants was not predictive of individual's recall performance,
recall similarity to their Free and Passive condition counterparts, semantic and causal centrality effects on memory,
nor their neighbor encoding effects (all ps>.3); however, higher percentage of choices granted predicted greater
individual tendency to forget the choice-denied events (
PE-boost (correlation between want-not and recall vectors per subject) vs percentage wanted:
Adventure: r(42) = 0.335, p = 0.026. Romance: r(49) = 0.137, p = 0.336.
see **Supplement S7** for details).



Together, these results suggest that in a context lacking full agentive control, perceived agency and their
effects on memory could vary across individuals in non-systematic ways. The one exception is that with more control
in such agency-uncertain contexts, the more one has reduced recall for the agency-denied events.