---
title: Interpreting Language Model Parameters
authors:
  - name: TODO authors (Approximately alphabetical for now before ordering for publication. Final pub will not be alphabetical)
    url: https://goodfire.ai
  - name: Lucius Bushnaq
    affiliation: Goodfire
    url: https://goodfire.ai
  - name: Dan Braun
    affiliation: MATS
    url: https://goodfire.ai
  - name: Bart Bussmann 
    affiliation: Goodfire
    url: https://goodfire.ai
  - name: Oliver Clive-Griffin
    affiliation: Goodfire
    url: https://goodfire.ai
  - name: Nathan Hu
    affiliation: Goodfire
    url: https://goodfire.ai
  - name: Michael Ivanitskiy
    affiliation: Goodfire
    url: https://goodfire.ai
  - name: Linda Linsefors
    affiliation: Goodfire
    url: https://goodfire.ai
  - name: Lee Sharkey
    affiliation: Goodfire
    url: https://goodfire.ai
  - name: tbd
    affiliation: Goodfire
    url: https://goodfire.ai

affiliation: Goodfire, MATS, tbd
correspondence: lee@goodfire.com
published: "April 2026"


---

Neural networks use millions to trillions of parameters to learn how to solve tasks that no other machines can solve. What structure do these parameters learn? And how does it compute intelligent behavior? 

Mechanistic interpretability aims to uncover how neural networks use their parameters to implement their impressive neural algorithms. Although previous work has uncovered substantial structure in the intermediate representations that networks use, little progress has been made to understand how the parameters and nonlinearities of networks perform computations on those representations. 

In this work, we present a method that brings us closer to this understanding. We decompose a language model's parameters into components that each implement a minimal amount of a network's computation, where as few components as possible are required to account for the network's behavior on any input. The method, adVersarial Parameter Decomposition (VPD), learns parameter components that reconstruct the language model's behavior while ablating as many parameter components as possible, including under ablations that are adversarially chosen to minimize reconstruction quality. Such ablatability, and their other qualities, makes these parameter components good candidates for mechanistically faithful, minimum length descriptions of a network's behavior, and generates accounts of network function on individual datapoints that should aggregate appropriately to more global descriptions of the network's learned algorithm. 

We study how sequences of interactions between these parameter components produce the network's output on particular inputs, enabling a new kind of 'circuit' analysis. While more work remains to be done to deepen our understanding of how neural networks use parameter components to compute their behavior, our work suggests an approach to identify a minimal set of simple, mechanistically faithful components on which further mechanistic analysis can be based.

## Introduction

<!-- Language models are remarkably intelligent. During training, their parameters learn to implement neural algorithms that we do not know how to design directly. -->

<!-- We can thus train machines to solve tasks that otherwise resist engineering solutions. However, since we did not design these neural algorithms ourselves, it means that a growing portion of our daily lives depend on systems that we do not deeply understand <cite>Bengio2026InternationalAISafety</cite>. -->

Mechanistic interpretability aims to reverse engineer neural networks, such as language models, so that we can understand the neural algorithms they have learned. Reverse engineering requires decomposing a system into simpler parts that we can study in relative isolation. Unfortunately, it is not obvious how best to decompose neural networks into such parts <cite>mueller2024questrightmediatorhistory, sharkey2025openproblemsmechanisticinterpretability</cite>.
The most straightforward candidates for these parts, such as neurons, attention heads, or whole layers, don't always map to individual, interpretable computations <cite>hinton1981parallel, wei2015understandingintraclassknowledgeinside, nguyen2016multifacetedfeaturevisualizationuncovering, olah2017feature, janiak2023polysemantic, jermyn2023attention, yun2021sparse, lindsey2024crosscoders</cite>.

Alternative approaches to decomposition, such as transcoders <cite>dunefsky2024transcodersinterpretablellmfeature, ameisen2025circuit</cite> or mixtures of linear transforms <cite>oldfield2025towards, lindsey2025molts</cite>, typically involve fitting a set of simple functions to the transitions between activations at different layers in the network, and linearly combining the outputs of these simple functions.
The idea here is to approximate the complex, nonlinear function implemented by the network's layers using a simpler, easier to understand function. These methods, sometimes called *activation-based decomposition* methods, have led to significant advances in our understanding of the intermediate representations inside neural networks when computing their outputs <cite>dunefsky2024transcodersinterpretablellmfeature, ameisen2025circuit</cite>.

But identifying representations is not the same as understanding the computations that use those representations as their inputs and outputs. Unfortunately, because the simpler functions that these methods use are of a different functional form to the original network, it is hard to relate their accounts of network function to the actual objects that are doing the computations, namely the network's parameters and nonlinearities.

This is not just a theoretical issue. It prevents us from achieving practical engineering goals. For example, it makes it challenging to know how to make precise, predictable modifications to a model's neural algorithm by making edits to its parameters. It also makes it hard to predict how the model's neural algorithm will perform on a different distribution to the one it was studied on.

The mismatch of functional form between models and their activation-based decompositions is an important issue, but it is not the only one: Activation-based methods have not yet yielded decompositions that exhibit a fully satisfactory level of mechanistic faithfulness <cite>ameisen2025circuit</cite>, and suffer from a number of other issues (See <cite>sharkey2025openproblemsmechanisticinterpretability</cite> for review).

<!-- Consequently, methods such as sparse dictionary learning \cite{yun2021sparse, Sharkey_2022, cunningham2023sparse, cunningham2023sparseautoencodershighlyinterpretable, bricken2023monosemanticity}, transcoders \cite{dunefsky2024transcodersinterpretablellmfeature, ameisen2025circuit}, and mixtures of linear transforms (MOLTs) \cite{oldfield2025towards, lindsey2025molts} were introduced to decompose datasets of neural activations, with the hope that they would identify units that approximate the network's underlying computational units. These methods, sometimes called \textit{activation-based decomposition} methods, unfortunately suffer from a range of issues, including feature splitting \cite{bricken2023monosemanticity, chanin2024absorptionstudyingfeaturesplitting} and unreliable level of mechanistic faithfulness \cite{ameisen2025circuit} (See \cite{sharkey2025openproblemsmechanisticinterpretability} for a more comprehensive review of these methods). Mechanistic unfaithfulness is suboptimal for activation-based methods' use in mechanistic analysis, and arises in part because these methods do not optimize for it. They approximate parts of the original network using functions of a different functional form as parts of the network. Their accounts of networks' computations are therefore not given in terms of the actual objects that are doing the computations---the network's parameters and its nonlinearities. -->

These issues motivate alternative approaches to mechanistic decomposition, including parameter decomposition methods <cite>braun2025interpretabilityparameterspaceminimizing, bushnaq2025spd, chrisman2025identifyingsparselyactivecircuits</cite>, which give accounts of network function in terms of the parameters that the network uses on each datapoint. *Ablation-based parameter decomposition methods* <cite>braun2025interpretabilityparameterspaceminimizing, bushnaq2025spd</cite> identify a set of parameter components where as few components as possible are "necessary" to perform the same computations original network on any datapoint, where "necessary" means that they cannot be ablated on a given datapoint without adversely affecting output reconstruction error<footnote>Including, crucially, partial ablations.</footnote>. Simultaneously, the parameter components are selected to implement as simple computations as possible and to sum collectively to the target network's parameters. If parameter components exhibit all these properties, then they are strong candidates for the network's 'ground truth' mechanisms<footnote>Though one would first need to accept philosophically that such mechanisms can be said to exist in non-toy networks!</footnote>.

Parameter decomposition methods can identify known ground truth mechanisms in toy models that: Are not necessarily aligned to architectural components such as neurons, individual attention heads, or layers; operate on representations in superposition; or are multidimensional. And, due to the requirement that components sum to the target model, parameter decomposition methods should not exhibit feature splitting. Notably, parameter decomposition methods can readily be applied to any architecture, unlike activation-based methods, where it has been challenging to use the same decomposition methods to decompose both attention layers and MLPs <cite>kamath2025tracing, ameisen2025circuit, wynroe2024decomposing, ge2024localglobal</cite>. In demonstration of this ability, previous work has used ablation-based parameter decomposition to identify induction heads in a transformer trained on a toy model of induction <cite>christensen2025decomposition</cite>.

Ablation-based parameter decomposition methods thus promise solutions to many of the issues of activation-based decomposition methods. However, prior parameter decomposition proposals have several important shortcomings, which we resolve in this work:

<!-- TODO(Lee) Deal with Nathan and Misha's feedback on this part of the intro -->

- **No application to full language models**: While the most recent parameter decomposition method, Stochastic Parameter Decomposition (SPD)<cite>bushnaq2025spd</cite> is more scalable than its predecessor, Attribution-based Parameter Decomposition <cite>braun2025interpretabilityparameterspaceminimizing</cite>, it has not yet been applied to full language models.
<!-- Oli: "I feel like the introduction of adversarial vulnerability needs more motivation here. I can imagine this line being a "huh?" moment for many people. having a think about this." https://goodfire-ai.slack.com/archives/C0ADGSRABDW/p1775040969484549?thread_ts=1775034924.912679&cid=C0ADGSRABDW
-->
- **Insufficient ablation robustness to ensure mechanistic faithfulness**: While some work has applied SPD to a single layer of GPT2-small <cite>christensen2025decomposition</cite>, no application of SPD so far has measured key metrics that would be necessary to ensure mechanistic faithfulness, such as having good output reconstruction loss even under adversarially chosen ablations (rather than under only stochastically chosen ablations).
- **Partial incompleteness of the method (No clustering from subcomponents to components)**: Previous implementations of SPD have also been partially incomplete: Attribution-based parameter decomposition <cite>braun2025interpretabilityparameterspaceminimizing</cite> decomposed networks into full vectors in parameter space which span all parameters in the model. But SPD decomposes them into rank-one matrices, which are limited only to single parameter matrices. A full implementation of SPD requires a *post hoc* clustering step to combine multiple rank-one matrices into full vectors in parameter space. But previous work left this clustering step implicit <cite>bushnaq2025spd</cite>.
<!-- - **No analysis of nonlinear interactions between components**: Previous work omitted analyzes of the nonlinear interactions between parameter components, which would be crucial for assessing how useful parameter decomposition methods are for interpretability. -->

In this work, we resolve these issues with a method called *ad**V**ersarial **P**arameter **D**ecomposition* (**VPD**)<footnote>Regrettably, the acronym APD was taken by our previous work, Attribution-based Parameter Decomposition!<cite>braun2025interpretabilityparameterspaceminimizing</cite></footnote>.

VPD builds heavily on the SPD method introduced by <cite>bushnaq2025spd</cite> but has several important modifications, which together make it more mechanistically faithful and scalable to larger models than decomposed in previous work. The primary difference between VPD and SPD is in the ablations. On each datapoint, both SPD and VPD sample from the space of possible partial ablations of parameter components in order to check whether those parameter components can be partially ablated in any combination, thus identifying whether they are "necessary" for that datapoint. However, where SPD samples from the space of partial ablations using *stochastic* samples from the space, VPD uses *adversarially chosen samples* (<ref>sec:opt-mech-faithfulness</ref>) <footnote>Both approaches are nonetheless designed to approximate what would happen if we could check *all* potential partial ablations.</footnote>. The core details of the method are discussed in <ref>sec:method</ref>.

We use VPD to decompose a small language model ($67$M parameters, four layers) trained on the Pile <cite>gao2020pile</cite>. We find parameter subcomponents that are highly interpretable (<ref>sec:param-comps-interpretable</ref>), both in terms of the dataset examples that they activate on and how they interact with other subcomponents to produce specific behaviors (<ref>sec:circuits</ref>). We compare the parameter components that we find to the objects found by other decomposition methods, such as per-layer and cross-layer transcoder (CLT) latents. We find that VPD achieves a better tradeoff between sparsity and reconstruction under standard training objectives and is more robust to mismatches between training and evaluation protocols compared to end-to-end trained methods (<ref>sec:decomp-model-behav-sim</ref>, <ref>app:vpd-sparsity-acc-tradeoff</ref>). VPD also has comparable interpretability (<ref>sec:param-comps-interpretable</ref>) and exhibits less feature splitting (<ref>sec:splitting</ref>) than activation-based comparisons. 
We develop attribution graphs that let us study the circuits that underlie some language model behaviors (<ref>sec:circuits</ref>).
<!-- Furthermore, we analyze the nonlinear interactions between parameter components (<ref>app:interactions-gis-vs-coact</ref>). We demonstrate that complex nonlinear interactions are rarer than would be expected by chance, despite not being a property our method optimizes for directly, suggesting that it reflects an underlying computational simplicity in the target model itself.--> 
Finally, we demonstrate we can use our understanding of the network's parameters to manually rewrite its neural algorithm for emoticon predictions (<ref>sec:model-editing</ref>).

We release a library for reproducing our experiments at <a href="https://github.com/goodfire-ai/param-decomp" target="_blank">https://github.com/goodfire-ai/param-decomp</a>.
<!-- DAN: Is it possible to rename the repo to vpd without screwing up anything? -->

## The core method: adVersarial Parameter Decomposition {toc: Method: adVersarial Parameter Decomposition}

<label id="sec:method"/>


In this section, we introduce ablation-based parameter decomposition methods from scratch and highlight key differences between VPD and prior methods in this class. Although our method, VPD, builds heavily on SPD <cite>bushnaq2025spd</cite>, the following explanation of VPD does not assume familiarity with SPD or its predecessor <cite>braun2025interpretabilityparameterspaceminimizing</cite>.

Our goal is to decompose a neural network into the *mechanisms* that it uses to compute its behavior. Its mechanisms are what it uses to take input activations, compute its hidden activations, and finally compute its output. We don't approach this goal with strong presuppositions of what a "mechanism" is. But we take for granted that a typical network doesn't use all of its mechanisms on every input (or, at least, it doesn't use all of its mechanisms by the same amount). If that were not the case, then networks could not be said to be *modular*, having distinct parts that do different things on different inputs. Without modularity, networks simply couldn't be decomposed into separable functional units.

One candidate for the network's mechanisms is the network's parameters. Like mechanisms, networks appear not to use all of their parameters simultaneously on every datapoint <cite>veit2016residual, zhang2022moefication, dong2023attention</cite>. This happens, for instance, when a network's parameters "read from" activation subspaces that are orthogonal to the activations on that datapoint, thus projecting the activations to zero, thereafter having no downstream causal effect. Alternatively, if the activations fail to "activate" a given ReLU neuron, the activation of that neuron is zero, thereafter having no downstream causal effect. However, the network's parameters are in fact a single vector in the network's parameter space, and do not have an obvious decomposition into parts. How should they be decomposed into parts that comprise the network's mechanisms?

On a high level, parameter decomposition methods use the idea that it should be possible, for a given datapoint, to identify the "subset" of the network's parameters that are necessary and sufficient for computing its output on that datapoint. That "subset" of parameters should contain all the mechanisms used by the network on that datapoint. If particular "subsets" of the network's parameters are repeatedly used together by different datapoints, then they may be part of the same mechanism. Parameter decomposition methods therefore aim to find particular "subsets" of the network's parameters that tend to be used together, where as few of them as possible are necessary and sufficient for computing the network's output on any input<footnote>We use the word "subset" loosely here. In practice, parameters are not divided into discrete sets. The network's parameters are a vector in parameter space, and we want some way to divide up that vector into 'parts' in a way that they still 'make up' the original parameters.</footnote>.<footnote>An analogy that is sometimes helpful for understanding VPD is that it is similar to Singular Value Decomposition on a weight matrix, except where we decompose the matrix into more components than the rank of the matrix, and where the components we identify are the parts of the matrix that have similar downstream causal effects on the data distribution, thereby taking downstream nonlinearities into account.</footnote>


More concretely: If particular parameters are unused by the network on a particular datapoint, then we should be able to ablate them (including partially) on that datapoint without adversely affecting the network's output. Ablation-based parameter decomposition methods thus aim to decompose network parameters into a set of vectors in parameter space called *parameter components*. Parameter components are trained to exhibit a number of specific properties such that, if they exhibit those properties, they would be good candidates for the network's "mechanisms". They are trained to be:

- **Parameter-faithful**: They sum to the network's total parameter vector;
- **Minimal**: As few components as possible are causally important for computing the network's output on any particular input;
- **Mechanistically faithful**: Every subset of components that includes the causally important components is sufficient to compute the network's output on any particular input;
- **Simple**: Each component should involve as little computational machinery as possible.

In the following sections, we define parameter components concretely and explain how they are optimized to exhibit each of these four properties.


### Parameter components consist of subcomponents {toc: Subcomponents}

<label id="sec:method-components"/>

Suppose we have a neural network $f(x;\theta)$ with parameters $\theta$. We would like to decompose this parameter vector into a sum of *parameter components* with the above properties. 

It would be computationally expensive to decompose models into whole parameter vectors, since each such vector would have a memory cost equivalent to the whole target model. Therefore, as in <cite>bushnaq2025spd</cite>, we use a less expensive way to parameterize parameter components: Although its parameters $\theta$ can be expressed as a single large vector, they are more commonly conceptualized as a set of matrices $\theta = \{W_1, \dots, W_L\}$. We further decompose individual matrices into sums of rank-one matrices called *subcomponents*, each parameterized as an outer products of two vectors: 

$$W_l \approx \sum_{c} \vec{U^l_c} (\vec{V_c^{l}})^\top = U^l (V^l)^\top , $$ 

where there may be more subcomponents than rows and columns in the matrix. Permitting more subcomponents than rows and columns in the matrix allows VPD to identify mechanisms that operate on representations in superposition<cite>Vaintrob_Mendel_Kaarel_2024, Bushnaq_Mendel_2024, elhage2022toy</cite>. 



<figure class="wide">
<label id="fig:sum_components"/>
<img src="figures/Sum of components.png">
<figcaption>Parameter decomposition methods decompose target model parameters into vectors in parameter space (parameter components) that are optimized to approximate the model's mechanisms. </figcaption>
</figure>

Although a single subcomponent *explicitly* parameterizes only a single weight matrix, it *implicitly* parametrizes a full parameter vector if we assume it takes values of $0$ in all other weight matrices. It is therefore possible to combine these subcomponents into full parameter components by adding them together in the right way. We identify these components using a subcomponent clustering method. Previous work left this clustering step implicit, but in this paper we introduce an explicit method (<ref>app:clustering</ref>).


### Enforcing parameter faithfulness with $\Delta$-components {toc: Enforcing parameter faithfulness}

<label id="sec:method-delta-components"/>

To ensure the components collectively sum to the parameter vector of the target model, we define additional Delta-components, $\Delta^l$, that parametrize the difference between our subcomponents and the original model's matrices:

```equation
label: eq:delta_l2
tex:
  \htmlClass{hc-dl-delta}{\Delta^l}
  :=
  \htmlClass{hc-dl-W}{W^{l}}
  -
  \htmlClass{hc-dl-summed}{
    \sum_{c}
    \htmlClass{hc-dl-uv}{\vec{U^l_c} (\vec{V_c^{l}})^\top}
  }
tips:
  - hc-dl-delta: The Δ-component for target model parameter matrix l
  - hc-dl-W: Target model parameter matrix l
  - hc-dl-summed: The summed parameter subcomponents
  - hc-dl-uv: Rank-1 parameter subcomponent c for matrix l
```

We also encourage the $\Delta^l$-components to be small with an auxiliary MSE loss ($\mathcal{L}_{\text{Delta-L2}}$) (<ref>sec:vpd_delta_l2</ref>).


### Optimizing for minimality

<label id="sec:opt-minimality"/>

We want as few subcomponents as possible to be causally important for computing the network's output on any particular input. We therefore need some way to estimate which parameter subcomponents are "necessary" for computing the network's output on a given datapoint. We also require a notion of how well the "necessary" subcomponents have reconstructed the network's output.

Ablation-based parameter decomposition methods contend that a parameter subcomponent is "necessary" if it cannot be ablated without affecting the model's output on that datapoint. As in <cite>bushnaq2025spd</cite>, we train a *causal importance function* to predict how ablatable each subcomponent is on each batch and sequence position. We also implement the causal importance function using a neural network, though we use a different architecture (<ref>sec:vpd_ci_function</ref>).

We call the output of this function the *causal importance values*, $g^l_{b,t,c}\in[0,1]$ (for each subcomponent $c$ of weight matrix $l$ at a given batch index $b$ and sequence position $t$):

- If $g^l_{b,t,c} = 0$, then we should be able to fully or partially ablate that subcomponent on the forward pass at position $b,t$ without affecting the final model output.
<!--  when multiplying the model's hidden activations with weight matrix $l$ -->

- If $g^l_{b,t,c} = 1$, then it should not be possible to ablate that subcomponent without affecting the model's output on that datapoint<footnote>The Delta components $\Delta^l$ should always be ablatable, so they are assigned a causal importance of $0$ by definition.</footnote>. 

We want as few subcomponents as possible to be required to compute the output, so we train the causal importance values $g^l_{b,t,c}$ to take minimal values with an *importance minimality loss*:

<label id="eq:minimal"/>
  
$$
\begin{aligned}
\mathcal{L}_{\text{importance-minimality}}
  =
  \frac{1}{BT}
  \sum^{B}_{b=1}
  \sum^{T}_{t=1}
  \sum^{L}_{l=1}
  \sum^C_{c=1}
  \vert g^l_{b,t,c} \vert^p,
\end{aligned}
$$

<!-- LaTeX original:
\label{eq:minimal}
\begin{aligned}
\mathcal{L}_{\text{importance-minimality}}=\frac{1}{XT}\sum_{x,t}\sum^L_{l=1}\sum^C_{c=1} \vert g^l_c(x,t) \vert^p,
\end{aligned}
-->


where $p>0$.<footnote>The $\Delta^l$-components are defined always to have causal importance values of zero, since they should never be "necessary" to compute the model output.</footnote>


### Optimizing for mechanistic faithfulness

<label id="sec:opt-mech-faithfulness"/>

Components and their causal importances should be mechanistically faithful to the original model. One way of operationalizing this is to insist that, on any given data point, it should ideally be possible to ablate all causally unimportant components from the model weights, using any combination of ablations, without changing the model output. Another, more succinct, way of saying this is that *every* subset of components that includes the causally important components should be sufficient to compute the network's output on any particular input.

This is a much stricter requirement than merely demanding that the output should be invariant to the joint ablation of all causally unimportant components together. To see why it is stricter, suppose that two components $\theta_A$ and $\theta_B$ can be *jointly* ablated, but not *individually* ablated, on a data point without affecting the output<footnote> One way this could happen if $\theta_A$ and $\theta_B$ cancel each other out by influencing the final model output vector in opposite directions.</footnote>. Then we would consider both $\theta_A$ and $\theta_B$ to be causally important on that datapoint, whereas the less strict criterion might consider them both causally unimportant because they happen to be jointly ablatable. In other words, the stricter criterion demands an unchanged model output over a whole set of points in parameter space, whereas the less strict one demands it only for a single point. For an illustration of why this stricter condition is neccessary, see <ref>sec:vpd_recon_motivation</ref>.

To check whether subcomponents are ablatable, we define ablation masks $m^l_{b,t,c}\in[g^l_{b,t,c},1]$ for each subcomponent at each batch index $b$ and sequence position $t$. So, if a subcomponent has causal importance $g^l_{b,t,c}=1$, the only permitted value for the mask $m^l_{b,t,c}$ is also $1$, whereas if the causal importance is $0$, its mask can take any value between $0$ and $1$. These masks define new weight matrices $W^{\prime l}_{b,t}$ which we should be able to insert in place of the original model matrices $W^l$ without substantially changeing the model's final output. 

We operationalize this by demanding that the KL-divergence $D$ between the model output on the original forward pass and on forward passes using the masked weights should be small:

```equation
label: eq:random_recon
tex:
  \begin{aligned}
  \mathcal{L}_{\text{masked-recon}}
  &=
  \frac{1}{B}
  \sum^{B}_{b=1}
  \htmlClass{hc-stoch_rec-divergence}{
    D
    \Big(
      \htmlClass{hc-stoch_rec-stoch_output}{
        f(
          x_b
          \vert
          \htmlClass{hc-stoch_rec-w_stoch}{
            {W'}^1_b(
              m^1
            ),\dots,{W'}^1_b(
              m^L
            )
          }
        )
      },
      \htmlClass{hc-stoch_rec-target_output}{
        f(
          x_b
          \vert
          \htmlClass{hc-stoch_rec-target_weight}{
            W^1,\dots,W^L
          }
        )
      }
    \Big)
  } \\
  \end{aligned}
tips:
  - hc-stoch_rec-divergence: The KL-divergence between the target model and the masked model.
  - hc-stoch_rec-stoch_output: The decomposed model's output on datapoint x_b
  - hc-stoch_rec-w_stoch: The weight matrix created by masking parameter components
  - hc-stoch_rec-target_output: The target model's output on datapoint x_b
  - hc-stoch_rec-target_weight: The target model's weights
```

Ideally, we would calculate this masked reconstruction loss for every permitted combination of ablation masks $m$ for all subcomponents<footnote>And Delta components.</footnote> in all the model's weight matrices, but this would require performing an intractably large number of forward passes. So we instead use ablation masks $m$ drawn with both

1. **Stochastic sampling**, with ablation masks $m^{\text{stoch}}$ drawn from uniform distributions. This yields the *stochastic reconstruction loss*, $\mathcal{L}_{\text{stochastic-recon}}$.
2. **Adversarial sampling**, using ablation masks $m^{\text{adv}}$ optimized via gradient ascent to maximise the reconstruction loss. This yields the *adversarial reconstruction loss*, $\mathcal{L}_{\text{adversarial-recon}}$.
 

For details on the stochastic and adversarial sampling, see <ref>sec:recon</ref>.


### Optimising for simplicity

<label id="sec:methods-simplicity"/>

Each component ought to contain as little computational machinery as possible. Otherwise, we could say that the target model is one big parameter component, and proclaim our decomposition as complete without doing any actual decomposition! 

Our subcomponents are rank-one, which does constrain them to be simpler objects than full matrices. Unfortunately, this is not enough of a constraint on simplicity, because some rank-one solutions are "simpler" than others. In some situations, it is possible to add multiple rank-one mechanisms parametrizing independent computational machinery used on disjoint subsets of the data together and have the resulting sum also be rank-one.
<footnote>A theoretically clean motivating example of this phenomenon is the toy model of ping pong superposition <cite>gibson2025</cite>. In the ping pong superposition construction, $64$ superposed rank $1$ circuits can be implemented in layers of width $21$. Only one circuit is ever active at a time, and groups of eight circuits each share the exact same origin or target neurons. Components for circuits in the same group can then be summed, and the result will again be exactly representable as a rank $1$ matrix, which is causally important for computing the output exactly when any of the circuits in the group are causally important for computing the output. Hence if we apply VPD to this toy model, the importance minimality loss alone will provide no incentive to further separate the eight rank $1$ matrices for the eight circuit groups into $64$ rank $1$ matrices for the $64$ individual circuits, leaving us with components that activate polysemantically and contain more computational machinery than they need to.</footnote>
<footnote> We observed indications that some VPD decompositions suffered from this failure mode. Sometimes, subcomponents seemed to be involved in multiple (usually two) unrelated computations, which depended on whether the incoming activations had strong positive or negative inner products with the subcomponent's right singular vector.</footnote> 

We therefore encourage breaking up subcomponents into multiple that are causally important on as few data points as possible by introducing an additional, slightly superlinear, penalty on subcomponent activation frequency:

<!--
<figure class="fig-simplicity">
<label id="fig:simplicity"/>
<img src="figures/simplicity.png">
<figcaption></figcaption>
</figure>
-->
<!-- DAN: I don't think this figure says enough to be worth it. If it also illustrated how two rank one subcomponents can add up to a rank one matrix, then it might be worth it.   -->
<!-- Oli: agreed, could illustrate this relatively easily in 2d or 3d I think. Also maybe important to make clear that having the same left or right SV doesn't mean you're doubling-up in parameter space -->

```equation
label: eq:freq_minimality
tex:
  \begin{aligned}
  \mathcal{L}_{\text{frequency-minimality}}
  =
  \frac{1}{B T}
  \sum^{B}_{b=1}\sum^{T}_{t=1}\sum^L_{l=1}\sum^C_{c=1}
  \htmlClass{hc-g-left}{\vert g^l_{b,t,c} \vert^p}
  \htmlClass{hc-g-right}{  
    \log_2(
    1 +
    \sum^{B}_{b'=1}\sum^{T}_{t'=1}
    \vert
      g^l_{b',t',c}
    \vert^p
  )},
  \end{aligned}
tips:
  - hc-g-left: This term is just the causal importance set to the p^th power, similar to the importance minimality loss
  - hc-g-right: This term sums over the batch and is therefore higher for higher frequency subcomponents
```


There are probably multiple ways to optimize for the computational simplicity of parameter components, and we are not confident this choice is optimal (nor the choices of the other losses). Nonetheless, we found it to work well enough in practice. See <ref>sec:vpd_frequency_penalty</ref> for a more detailed motivation of this loss.

<!--<footnote>The more often a component is causally important, the more precisely we need to define it to obtain low total reconstruction over $X$ data points. If it is always causally important, every bit of precision in the component's definition we get wrong can hurt our loss on every data point. If it is rarely causally important, every bit we get wrong will only hurt our reconstruction loss on some fraction $\frac{\sum_{x,t} \vert g^l_c(x,t)\vert^0}{XT}$ of the data. Thus, we can afford to store it to fewer bits of precision without increasing reconstruction KL divergence too much. Under some simplifying modeling assumptions, this trade-off scales with $\log_2(\frac{\sum_{x,t} \vert g^l_c(x,t)\vert^0}{XT})$. We approximate the $L^0$ norm with an $L^P$ norm, and add $1.0$ to the argument for stability, since $p$-norms with $0<p$ can otherwise take values below $1.0$ and turn the $\log_2$ term negative.</footnote>.-->


<!-- Nonetheless, we remark that the new loss introduces an interesting symmetry with the importance minimality loss: 

- $\mathcal{L}_{\text{importance-minimality}}$ encourages *datapoints* in the training set to activate as few *subcomponents* as possible; minimizing it incentives de/compositions to employ *fewer* components.
- $\mathcal{L}_{\text{frequency-minimality}}$ encourages *subcomponents* to activate on as few *datapoints* in the training set as possible; minimizing it incentives decompositions to employ *more* components in total (at least relative to the $\mathcal{L}_{\text{importance-minimality}}$ loss).

We note that this tradeoff avoids creating the same problem as "feature splitting" due to the constraints imposed by the adversarial and stochastic reconstruction loss (<ref>sec:splitting</ref>).-->
<!-- \textit{(TxDx potentially: figure illustrating the effects of the two losses on causal importances)(TxDx experiment: showing CIs on real data, where y axis is C and data index is on X axis, where the C activations are they're hierarchically clustered - should show that each component activates on fewer parameter components and there are fewer 'bias-like' components).} -->

### Summary of loss terms

<label id="sec:methods-summary"/>

In total, our loss function has five terms:

$$
\begin{aligned}
\mathcal{L}_{\text{VPD}} ={}
  & \beta_1 \mathcal{L}_{\text{adversarial-recon}} \\
  + & \beta_2 \mathcal{L}_{\text{stochastic-recon}} \\
  + & \beta_3 \mathcal{L}_{\text{importance-minimality}} \\
  + & \beta_4 \mathcal{L}_{\text{frequency-minimality}} \\
  + & \beta_5 \mathcal{L}_{\text{Delta-L2}}
\end{aligned}
$$

<!-- LaTeX original:
\begin{aligned}
\mathcal{L}_{\text{VPD}}=\mathcal{L}_{\text{adversarial-recon}}+\mathcal{L}_{\text{stochastic-recon}}+\mathcal{L}_{\text{importance-minimality}}+\mathcal{L}_{\text{frequency-minimality}}+\mathcal{L}_{\text{Delta-L2}}
\end{aligned}
-->

They each optimize the parameter components to exhibit particular properties: 

- The $\mathcal{L}_{\text{adversarial-recon}}$ and $\mathcal{L}_{\text{stochastic-recon}}$ losses optimize for **mechanistic faithfulness**. (<ref>eq:random_recon</ref>)
- The $\mathcal{L}_{\text{importance-minimality}}$ loss optimizes for **minimality**. (<ref>eq:minimal</ref>)
- The $\mathcal{L}_{\text{frequency-minimality}}$ loss optimizes subcomponents for **simplicity**. They are also constrained to be rank-1 matrices, which imposes one aspect of simplicity. (<ref>eq:freq_minimality</ref>)
- The $\mathcal{L}_{\text{Delta-L2}}$ auxilliary loss optimizes for **parameter-faithfulness**, even without the $\Delta$-components, which ensure it. (<ref>eq:delta_l2</ref>)

The key difference between VPD and our previous work <cite>bushnaq2025spd</cite> is the $\mathcal{L}_{\text{adversarial-recon}}$ and $\mathcal{L}_{\text{frequency-minimality}}$ losses. There are several other, smaller differences that do not fundamentally change the method but that we found helpful for decomposing language models. For more details, see <ref>sec:vpd_methods</ref>.

We evaluate the quality of our decomposition on a number of key metrics. For assessing the quality of a decomposition, the most important are $\mathcal{L}_{\text{adversarial-recon}}$ and $L_0$ per datapoint. For readers looking for practicable advice on how to tune hyperparameters and key optimization metrics, we provide a detailed *Training recipe for VPD* in <ref>app:recipe</ref>.

## Analyzing language model parameter subcomponents {toc: Analyzing subcomponents}

### Target language model

<label id="sec:langauge-model-details"/>

<!-- We trained a four-layer, 67M parameter decoder-only transformer model on an uncopyrighted subset of The Pile \cite{gao2020pile}. It uses standard multihead attention layers \cite{vaswani2017attention} with RoPE positional encoding and MLPs with a GELU activation function. RMSNorm is applied to the inputs of the attention and MLP layers \cite{xiong2020layernormalization} and before the final unembedding layer. The token embedding and LM head weights are tied, giving the model approximately 28M non-embedding parameters and 67M total parameters. The model achieves a final validation cross-entropy loss of approximately 2.71. The model architecture is summarized in Table \ref{model-hyperparams}; full training details can be found in Appendix \ref{app:training-details}. -->

We trained a four-layer 67M parameter decoder-only transformer model on an uncopyrighted subset of The Pile <cite>gao2020pile</cite>. A summary of the model architecture and training results can be found in <ref>tab:model-hyperparams</ref> and full training details of our target model can be found in <ref>app:training-details</ref>.


<figure>
<label id="fig:placeholder"/>
<img src="figures/transformer_diag.png">
<figcaption>Our target model is a standard decoder-only transformer language model.</figcaption>
</figure>

<figure> 
<label id="tab:model-hyperparams"/>

| Attributes of our target model |  |
|---|---|
| Layers | 4 |
| Residual stream dimension | 768 |
| MLP intermediate dimension | 3072 |
| Attention heads | 6 |
| Attention head dimension | 128 |
| Context length | 512 |
| Vocabulary size | 50,277 |
| Positional encoding | RoPE <cite>su2024roformer</cite> |
| Normalization | RMSNorm <cite>zhang2019rootmeansquarelayer</cite> |
| Activation function | GELU <cite>hendrycks2016gelu</cite> |
| Attention type | Standard Multi-Head Attention <cite>vaswani2017attention</cite> |
| Tied embeddings | Yes |
| Non-embedding parameters | ~28M |
| Total parameters (incl. embedding) | ~67M |
| Training dataset | The Pile <cite>gao2020pile</cite> (subset) |

</figure>

We decomposed the 24 weight matrices in this model<footnote>Except for the embedding and unembedding matrices.</footnote> into a total of $38912$ rank $1$ subcomponents, though the training only ended up using about $10000$. The others are essentially dead, having mean causal importances below $10^{-6}$. On average, each datapoint uses 205 subcomponents per sequence position, representing 2.1% of all alive components. The table below displays per-layer summary statistics for the decomposition.

<figure> 
<label id="tab:num-components-per-layer"/>

|  Layer  |  $C$    | Alive  | Mean L0 | L0/Alive |
|---------|---------|--------|--------------|--------------|
| Layer 0 | $9728$  | $3709$ | $44.6$       | $0.012$       |
| Layer 1 | $9728$  | $848$  | $18.9$       | $0.022$       |
| Layer 2 | $9728$  | $1943$ | $49.5$       | $0.025$       |
| Layer 3 | $9728$  | $3472$ | $92.0$       | $0.026$       |
| Total   | $38912$ | $9972$ | $205.0$      | $0.021$       |

<figcaption> Per-layer decomposition summary statistics: Subcomponent dictionary sizes $C$; alive subcomponents (subcomponents with mean causal importances above $10^{-6}$ at the end of training); average $L_0$ scores of subcomponents with causal importance $>0$ per batch and sequence position; and fraction of all subcomponents with causal importance $>0$ per batch and sequence position. </figcaption>
</figure> 


### The decomposition model behaves similarly to the target model {toc: Decomposition model behavior}

<label id="sec:approx-target-well"/>


If a decomposition method has correctly identified the mechanisms underlying a model's computation, then activating only the mechanisms that the method identifies as causally important on a given input should approximately reproduce the model's behavior on that input. Conversely, if a replacement model fails to reproduce the model's behavior, then the decomposition has either missed important mechanisms or identified spurious ones. Reconstruction quality is therefore a necessary (though not sufficient) condition for a decomposition to be mechanistically faithful.

Our parameter components capture different amounts of the target model's performance depending on how masks are calculated (<ref>tab:vpd-ce-compute-compar</ref>). One quantitative measure of performance is cross-entropy (CE) loss on the validation set: the decomposed model achieves between 2.72 and 3.02, compared with 2.71 for the target model.


These seem quite similar, indicating that the decomposition has captured many of the mechanisms that the target model uses to perform well on the validation set. 

That said, it is important to contextualize these numbers relative to some baseline. A metric that is sometimes helpful is *Pretraining Compute Recovered*<cite>gao2024scalingevaluatingsparseautoencoders</cite>, which is the percentage of the target model's total pretraining compute at which the target model's training curve reaches the same validation CE loss as the reconstruction model (i.e. a value of X% means the reconstruction performs no better than the target model did when only X% of pretraining was complete). When we exclude the $\Delta$-component (which is trained to be as causally unimportant as possible), the remaining *unmasked* parameter components recover about $82\%$ of the pretraining compute. When using stochastic ablations, this drops to around $27\%.$

<label id="tab:vpd-ce-compute-compar"/>
| Masking mode (excluding $\Delta$-components) | Validation CE Loss | Pretraining Compute Recovered (%)  |
|:---------------------------------------------|-------------------:|:--------------------------------|
| **Target Model**                             |    **2.71**        | **100%**                        |
| Unmasked (All masks$=$1)                     |      2.72          | 82.4%                           |
| Stochastic Masks                             |      2.84          | 26.9%                           |
| Rounded Masks (Mask$=$1 if CI$>$0)           |      2.94          | 11.8%                           |
| Rounded Masks (Mask$=$1 if CI$>$0.1)         |      2.95          | 11.3%                           |
| Causal Importance values (CIs) used as Masks |      2.99          | 9.4%                            |
| Rounded Masks (Mask$=$1 if CI$>$0.5)         |      3.02          | 8.0%                            |


Pretraining compute recovered is an important metric, but it is rarely reported, so comparisons to other methods are difficult. Nonetheless, VPD compares favorably to the only other method in the literature that we are aware of that reports this metric: Top-$k$ SAEs <cite>gao2024scalingevaluatingsparseautoencoders</cite> reports a pretraining compute recovered of $10\%$ when replacing a single layer of GPT-4 with an SAE with 16 million latents. By comparison, even though our approach decomposes the whole model rather than just a single layer, it recovers between $8\%$ and $27\%$, depending on the ablation method used.

The table below shows KL-divergence to the target model under adversarial masking with different numbers of adversarial optimization steps, calculated across a batch of $128$ of sequence length $512$ drawn from the evaluation set. The adversarial masks were calculated with Projected Gradient Descent (PGD)<cite>madry2018towards</cite> optimization, sharing the same source for each component across the batch. For more details on the PGD loss evaluation metric see <ref>sec:vpd_methods-adv</ref>.

<label id="tab:vpd-pgd-ce"/>
|  Adversarial optimization steps $n^{\text{adv}}$ | KL divergence to target model | 
|:-------------------------------|----------:|:-----------------------------|
| 20                 | 0.8280 	|
| 40         		     |      1.3539 	| 
| 80                 |      3.8381 	| 
| 160		             |      25.2560 	|
| 320          	     |      40.2200	 |
While the decomposition is somewhat robust to approximately $20$ steps of adversarial optimization, it is clearly not at all robust to $160$ steps or more.<footnote> To provide some sense of scale, zero-ablating all of the target model's weight matrices we decomposed gives a KL-divergence of ca. $67.2449$.</footnote> For some discussion on how much adversarial robustness a decomposition should have to be useful for model editing and interpretability in practice, see <ref>sec:discussion-robustness</ref>.
<!--TODO(Dan)(Low priority) KL div. vs. random uniform distribution over tokens would be neat to give as a reference point here as well-->
<!--TODO(Dan)(Low priority): It'd be nice to show how the adv robustness scales with the batch size too.  -->

Qualitatively, the generations produced by different sampling methods align with these quantitative measures. The generations seem qualitatively to produce similar behavior to the target model in most cases (<ref>fig:generations-showcase</ref>). Surprisingly, even when masks are adversarially sampled, the generations are not *entirely* nonsensical. This is feasible because we only get to adversarially sample causally unimportant parameter components. 




<label id="fig:generations-showcase"/>
  
```generations
data: data/generation_comparisons.json
caption: Side-by-side generation comparisons across masking strategies.
```

<!-- TODO(Dan, Lee)(Low priority) re write validation loss measurement script so it can do adversarial too -->


### VPD has a better tradeoff between reconstruction versus sparsity compared with transcoders {toc: Comparison with transcoders}

<label id="sec:decomp-model-behav-sim"/>

Any decomposition of a neural network faces a fundamental tradeoff between the number of `objects' they use to reconstruct the network's behavior and the quality of that reconstruction. If a decomposition can use fewer objects to capture the same amount of network performance, then that explanation is preferred according to Occam's razor, assuming the objects use a similar amount of computational machinery.


We study the reconstruction versus sparsity tradeoffs of different decompositions and compare the VPD model with two families of activation-based decomposition methods: Per-layer transcoders (PLTs) <cite>dunefsky2024transcodersinterpretablellmfeature</cite> and cross-layer transcoders (CLTs) <cite>lindsey2024crosscoders</cite>, both using BatchTopK <cite>bussmann2024batchtopk</cite>. We simultaneously replace all 4 MLP layers of the target model with their sparse reconstructions and measure the resulting increase in cross-entropy loss relative to the unmodified target model. 

There isn't a straightforward apples-to-apples comparison between transcoder latents and VPD subcomponents, so we present a number of different comparisons (with more extensive experimental details in <ref>app:vpd-sparsity-acc-tradeoff</ref>) <footnote>Comparing sparsity across methods requires care, because each method has structurally different notion of what constitutes a single active component. A CLT feature writes to the residual stream at every layer simultaneously, while a PLT latent affects only one layer. VPD subcomponents are scoped to individual weight matrices, and each MLP layer has two such matrices (up-projection and down-projection).</footnote>. To ensure our conclusions are not artifacts of how we count components or latents, we show results under three possible definitions of sparsity: 

1.  **Average active components per module**: Active encoder latents for PLTs/CLTs; active subcomponents per weight matrix for VPD; 
1. **Active components per MLP output reconstruction**: Adjusting for the fact that a CLT latent affects multiple layers and that VPD uses two modules per MLP;
1. **Total active parameters**: VPD's rank-one subcomponents have more parameters than a PLT latent and a single CLT latent has multiple decoder vectors.

We compare VPD with PLTs and CLTs trained with their standard training losses, noting these are different objectives (VPD trains on output reconstruction while PLTs and CLTs are trained to reconstruct activations at each layer). 

<figure class="wide">
<label id="fig:pareto-mse"/> 
<img src="figures/pareto_mse_v4.png">
<figcaption>CE degradation when simultaneously replacing all 4 MLP layers with sparse reconstructions from each method. **(a)** Active components per module (raw L0). **(b)** Active components per MLP reconstruction, adjusting for CLT's cross-layer writes and VPD's paired modules. **(c)** Total active parameters. VPD (purple markers) Pareto-dominates the activation-based methods under all three sparsity measures. The dashed line indicates zero-ablation (all MLP outputs set to zero). Lower is better.</figcaption>
</figure>

We observe that VPD performs favourably compared with activation-based decomposition, achieving less CE degradation for a given $L_0$ across all three definitions of sparsity.

We noted above that VPD and the transcoders differ in training objective. VPD is trained end-to-end, wheras activation-based approaches are usually trained layerwise. This complicates direct comparison and arguably makes the above analysis somewhat unfair to activation-based methods. We address this by also comparing under matched objectives in <ref>app:vpd-sparsity-acc-tradeoff</ref> and find that VPD compares favourably to other methods: When trained and evaluated on a range of objectives, VPD's pareto domination disappears, but it avoids overfitting to its particular training objective, unlike the activation-based methods.

Additional figures and training logs for the VPD decomposition can be found at the WandB link <a href="https://wandb.ai/goodfire/spd/runs/s-55ea3f9b" target="_blank">here</a>.



### Parameter subcomponents are highly interpretable {toc: Highly interpretable}

<label id="sec:param-comps-interpretable"/>

In order to study a parameter component's role in the network's neural algorithm, we need a definition of what it means for it to be 'active' on a given datapoint. 

There are at least two reasonable definitions:

1. **Causal importance**: The causal importance function is trained to output a value between $0$ and $1$ that tells us exactly how important a particular subcomponent is on a datapoint. It tells us if the subcomponent is 'necessary' or 'required' or 'used' on that input. In many ways, this is a perfect definition of 'active'! However, it is not a 'local' measure of a subcomponent's activation: A subcomponent with a small causal importance value might interact strongly with the activations at a layer, only for its effect to be suppressed later by others. For a more `local' measure, we use the next definition.
1. **Subcomponent activation**: We define the subcomponent activation as $$a_c^l = ||\vec{U^l_c}|| (\vec{V^l_c})^\top \vec{h^l},$$ where $\vec{h^l}$ are the model's hidden activations before matrix $l$ <footnote>We multiply by $||\vec{U^l_c}||$ because neither the $\vec{U}$ or $\vec{V}$ vectors are normalized by default, and we therefore need to multiply by this norm to make their subcomponent activations comparable.</footnote>. This defines how much the activations interact with a given subcomponent, even if that interaction ultimately ends up not being causally important for the output. Due to superposition <cite>olshausen1997sparseovercomplete, goh2016decoding, elhage2022toy, Vaintrob_Mendel_Kaarel_2024, Bushnaq_Mendel_2024</cite>, there will be more interactions in general than there are causally important interactions.

Throughout this paper, we use both definitions, highlighting which type of activation we mean in each instance.

We find that parameter subcomponents tend to 'activate' (in both senses) for coherent categories of inputs. <ref>fig:components-showcase</ref> shows some dataset examples on which each component is causally important. It also shows the component activation in the underlines. You can navigate the panel to explore the activations of a variety of parameter subcomponents:


<label id="fig:components-showcase"/>

```components
data: data/model-overview
layer: 2.attn.k
caption: Browse all VPD parameter components by weight matrix. Green highlights indicate causal importances; colored underlines show subcomponent activations.
```

<!-- DAN: (Oli) I think the subcomponent indices should be labelled here so people can find these components in other views of our decomposition -->

To compare how 'interpretable' parameter components are to transcoder latents, we can measure how semantically coherent a component's activation patterns are using *intruder detection* <cite>chang2009reading, paulo2025evaluating</cite>. In intruder detection, we present an LLM-judge with a set of inputs that activate a given VPD component or transcoder latent alongside one 'intruder' example that does not activate it. We task the LLM-judge to identify the intruder example. It should be easier to identify the intruder among a more semantically coherent set of inputs. In the VPD setting, we use causal importance values in place of activation magnitudes and select intruder examples with similar activation densities. 

We find VPD intruder detection scores improve drastically when using CI values thresholded with 0.1, which filters low-CI noise <ref>fig:intruder-score</ref>. We think that filtering out small causal importances is justifiable: <ref>tab:vpd-ce-compute-compar</ref> shows 0.1-rounded performance has essentially the same performance as 0.0-rounded performance, suggesting that very little performance is captured by components with small activations. 

We observe that 0.1-rounded VPD components score competitively with CLTs and PLTs trained using a local (layerwise) MSE activation reconstruction loss <ref>fig:intruder-score</ref>. VPD components are more coherent than PLTs and CLTs trained end-to-end. 
  
<figure>
<label id="fig:intruder-score"/>
<img src="figures/intruder_score_bar_chart_clean.png">
<figcaption>Intruder detection scores for various CLT and PLT latents, and VPD subcomponents at different CI thresholds. Error bars are 95% bootstrap CIs on the mean. Dashed line is random chance accuracy (20%). Higher is better.</figcaption>
</figure>
<!--TODO (Oli): The caption says there are error bars, but I (Dan) can't see them in the figure  -->

### VPD suffers less from feature splitting {toc: Less feature splitting}

<label id="sec:splitting"/>

<!-- GRAVEYARD: *Feature splitting* is a well-known issue of activation-based dictionary learning methods such as PLTs, SAEs, or CLTs: When we train dictionaries of larger and largers sizes on the same activations, the model can allocate multiple near-duplicate features to represent the same underlying concept. This improves sparsity and reconstruction, but doesn't necessarily discover more distinct mechanisms. In the extreme, a transcoder could assign a unique latentto every individual datapoint in the training set, effectively memorizing the dataset rather than uncovering reusable, general patterns. Feature splitting inflates the number of latents that must be inspected while adding little new mechanistic insight. -->

<!-- GRAVEYARD: VPD requires the parameter components to sum to the targer model's parameters. This constraint makes it much harder for the decomposition to represent the same mechanism multiple times with distinct components: Any two near-duplicate parameter components would simply sum to a $2\times$ larger version of the underlying mechanism, which would have bad parameter-faithfulness (or perform badly on other losses, depending on how the decomposition compensated). Thus, VPD excludes arbitrary redundancy among components and should not exhibit 'splitting' behavior. -->



Feature splitting is a well-known issue in activation-based dictionary learning methods such as PLTs, SAEs, and CLTs <cite>chanin2024absorptionstudyingfeaturesplitting, Bricken_2023_dictionary</cite>. As dictionary size increases, these methods can improve sparsity and reconstruction by replacing a 'broad', reusable latent with several narrower, more context-specific ones. In the extreme, a transcoder could assign a unique latent to every individual datapoint in the training set, effectively memorizing the dataset rather than uncovering reusable, general patterns. 

VPD is less susceptible to this failure mode. The key reason for this is that subcomponents marked as causally unimportant are required to be ablatable in any combination, not just all simultaneously. The model therefore needs to be robust to variations in parameter space along the directions of these components for all batches and sequence positions, not just the ones on which they are causally important. Without this constraint, the decomposition might be able to invent overly 'narrow', context-specific components that do not actually exist in the computational structure of the original model but that sparsely activate while reconstructing the model's behavior on some narrow subset of the data. Robustness to adversarial ablations is thus a key constraint that prevents feature splitting by optimizing for mechanistic faithfulness. See <ref>sec:vpd_recon_motivation</ref> for further discussion.

<!-- A causal importance of $0$ merely means that we *can* ablate a subcomponent while maintaining performance, not that we *have* to.  -->
<!-- This prevents the method from improving sparsity by inventing overly 'narrow', context-specific components that do not actually exist in the computational structure of the original model, since . -->

<!-- This requirement ensures that components do not interfere with the computations on data points that they are not causally important on -->
<!--This ensures that  Gratuitous splitting into components that share underlying mechanisms hard. Because components must sum to the target model's parameters, splitting a mechanism into pieces means no single piece retains the full mechanism's parameters. To survive the adversary, VPD would therefore be forced to keep all the split pieces active simultaneously. However, because the importance minimality loss penalizes the total number of active components, this artificial co-activation actively worsens the loss. Thus, VPD is disincentivized from inventing extra components merely to fill capacity, splitting mechanisms only when the target model contains parameter pieces that can support reconstruction independently under ablation.-->

To test empirically whether VPD does avoid feature splitting, we incrementally increase the number of subcomponents used by different VPD runs and count the number of "alive" subcomponents (subcomponents that activate at least once every 1M tokens). We train VPD at four capacity levels corresponding to $0.5\times$, $1\times$, $2\times$, $4\times$ the subcomponent count of the main decomposition we study. We compare against PLTs and CLTs at 4k and 32k dictionary sizes.


<figure class="fig-simplicity">
<label id="fig:feature_splitting"/>
<img src="figures/feature_splitting.png">
<figcaption>Number of alive subcomponents as a function of total component capacity. PLTs and CLTs scale roughly linearly with dictionary size, staying close to the $y = x$ line. VPD (purple) remains flat at ~6,500-7,000 alive subcomponents regardless of capacity, indicating that additional capacity is not used for feature splitting. Dashed line: $y = x$ (all components alive).</figcaption>
</figure>

<ref>fig:feature_splitting</ref> shows that, unlike PLTs and CLTs, increasing VPD's capacity does not increase the number of subcomponents that the method actually uses, suggesting that feature splitting is not a significant problem for VPD. In <ref>app:confirming-feature-splitting</ref>, we confirm that our PLTs and CLTs are indeed splitting features rather than discovering genuinely new ones.



## Decomposing attention behaviors that are distributed across attention heads {toc: Decomposing attention}

<label id="sec:attn-analysis"/>

Transformer language models are significant in large part because they were the first architecture that enabled scalable sequence modelling. The crucial component that lets transformers perform computations across sequences is the attention layer (<cite>vaswani2017attention, Bahdanauetal2014</cite>. Attention layer computations posed a novel challenge for interpretability.

In previous work that studies attention layer computations, attention heads typically have been the primary units of analysis to study attention behaviors <cite>vig2019multiscalevis, clark2019doesbertlookat, elhage2021mathematical, olsson2022incontextlearninginductionheads, wang2022interpretability, janiak2023polysemantic, nam2025causalheadgatingframework</cite>. Unfortunately for interpretability, it is possible for attention layers to perform computations in a way that is distributed across multiple heads <cite>jermyn2023attention, jermyn2025attention</cite><footnote>This phenomenon is sometimes called 'attention head superposition'. However, we prefer to reserve that term for the specific case where the attention layer implements more computations than the number of heads it distributes them across, which might not happen in general.</footnote>. It would therefore be ideal if our decomposition methods could cope with attention computations that are distributed across heads. So far, it has been difficult to find satisfactory activation-based decomposition methods that can do this <cite>jermyn2025attention, mathwin2024gated, wynroe2024qkbilinear, Kissane_Conmy_Nanda_2024, kamath2025tracing</cite>. 

Fortunately, parameter decomposition methods offer some hope: As we've seen in <ref>sec:param-comps-interpretable</ref>, parameter subcomponents seem to decompose the parameters into specialized functional units. And since parameter components are vectors in parameter space, they therefore can span multiple attention heads!

In this section, we demonstrate that parameter components in attention layers are indeed interpretable, and can span multiple attention heads (and usually do!). Focusing primarily on attention layer 1, we study two attention layer behaviors ('Previous token behavior' and 'Previous syntactic boundary movement') and show how parameter components distribute these computations across heads. 

### Attention layer parameter subcomponents have specific interpretable roles {toc: Subcomponents have interpretable roles}

First, we look at a few parameter subcomponents attention layer 1. In this layer VPD identifies different numbers of parameter subcomponents in the Q (15), K (48), V (226), and O (97) projection matrices.<footnote>At mean ci cut-off $10^{-6}$.</footnote>

There are many interesting components that correspond to easily interpretable behaviors:

- <comp key>1.attn.q:308</comp> activates on tokens related to existence or the verb 'to be' and other 'copula' verbs.
- <comp key>1.attn.k:485</comp> activates on words that predict 'copula' verbs, such as `·there` or `·it` in "there is/it is".
- <comp key>1.attn.k:218</comp> activates on the the word `·it` (including capitalized variations and variants both with and without a leading space)
- <comp key>1.attn.k:119</comp> activates on punctuation, spaces, brackets, newlines and other 'interstitial' words.
- <comp key>1.attn.k:290</comp> activates on newlines and end-of-text tokens only.
- <comp key>1.attn.v:42</comp> activates on coordinating conjunctions, like `·and`, `·or`, `·but` and `·&`.
- <comp key>1.attn.v:178</comp> activates on words related to position in time and, to a lesser extent, space, like `·December`, `·South`, `·2002`, `·long` and `·far`.
- <comp key>1.attn.o:983</comp> Activates on the introductions or titles of texts, particularly scientific papers.


Additionally, there are some components whose role seems more related to 'sequence position' than having a particular semantic meaning:

- <comp key>1.attn.q:149</comp> and <comp key>1.attn.q:497</comp> tend to activate on the tokens immediately following the first token of the sequence (and, incidentally, reveal some of the shortcomings of our autointerp labelling method, which seems to have missed this!).
- <comp key>1.attn.k:315</comp>, <comp key>1.attn.k:357</comp> and <comp key>1.attn.k:121</comp> tend only to be causally important on the first few tokens of a sequence, though with some exceptions.

Together, these interpretations are encouraging, because they suggest that our decomposition is identifying parts of the network that are specialized for particular functional roles. 

### Attention layer parameter subcomponents typically span multiple heads {toc: Subcomponents span multiple heads}

We've seen evidence that attention components are specialized for specific semantic and computational behaviors. Now we investigate whether these subcomponents are 'located' in particular heads. 

In our model, the $W_Q$, $W_K$, $W_V$, and $W_O$ matrices are concatenated across attention heads. But we can easily split them into the matrices belonging to individual heads. Even though parameter subcomponents by default span all heads in a layer, most of their 'mass' could be localized in single heads if their weights in all heads were zero except in one. But if their parameters have nonzero norm in multiple heads, then this is weak evidence that they perform computations across multiple heads. 

We'll focus on the $W_K$ and $W_Q$ matrices for now (<ref>fig:qk_comp_weight_norm</ref>). We see that, in fact, most components have nonzero weight norm across each head, suggesting that most $W_K$ and $W_Q$ components might perform computations in a distributed way!


<figure>
<label id="fig:qk_comp_weight_norm"/>
<img src="figures/layer1_qk_combined.png">
<figcaption>The norm of the weights of each parameter component in each head. No parameter component is exclusively localized in a single head, suggestive of computations that are distributed across attention heads.</figcaption>
</figure>

<!-- TODO(Lee)(Low priority): Appendix figure for fig:qk_comp_weight_norm but instead for OV weights -->

While this is suggestive of distributed computations, it is only indirect evidence. We would need to understand the computations themselves in order to confirm that the computations are indeed distributed across heads. To do this, we will need new analysis tools. And we can make the problem slightly easier by separately studying the two main parts of the attention layer: The QK circuit and the OV circuit <cite>elhage2021mathematical</cite>. We'll focus on the QK circuit first.


### The QK circuit consists of interactions between pairs of parameter subcomponents {toc: The QK circuit}

In attention layers, $W_Q$ and $W_K$ matrices transform activations in the (normed) residual stream to make queries ($q = W_Q x$) and keys ($k = W_K x$) for all heads. We can split them into the keys and queries for each head (e.g. $q = [ W_Q^{1} x , \cdots , W_Q^{H} x]$). 

The attention scores of head $h$ is calculated as $Z^h = x^\top W_Q^{h \top} W_K^h x$, which are used to calculate the head's attention pattern, $A^h = \text{softmax} (x^\top W_Q^{h \top} W_K^h x) $. 

Although the $W_Q$ and $W_K$ matrices are usually represented as separate matrices, it is convenient to study them together as a single matrix, $W_{QK}^h = W_Q^{h \top} W_K^h$ <cite>elhage2021mathematical</cite>.


Prior to parameter decomposition, it was not obvious how best to further decompose this circuit into specialized functional units. But VPD decomposes the $W_Q$ and $W_K$ matrices in a sum of functionally specialized rank-one parameter subcomponents <footnote>The $V$ matrices of the subcomponents do not need $h$ indices because they only read from the residual stream. The $U$ matrices project into query or key space, and hence need $h$ indices.</footnote>: 

$$
W_Q^h = \sum_c  \vec{U_{Q,c}^{h}} (\vec{V_{Q,c}})^\top \qquad  \qquad W_K^h = \sum_c  \vec{U_{K,c}^{h}} (\vec{V_{K,c}})^\top 
$$

These subcomponents are secretly also a decomposition of the QK circuit:

<!-- Todo(Dan)(Low priority) maybe give this equation hover labels? -->

<label id="eq:qk-interactions"/>
$$
\begin{aligned}
W_{QK}^h &= W_Q^{h \top} W_K^h \\
&= \left( \sum_c \vec{U_{Q,c}^{h}} (\vec{V_{Q,c}})^\top \right)^\top \left( \sum_{c'} \vec{U_{K,c'}^{h}} (\vec{V_{K,c'}})^\top \right) \\
&= \sum_{c, c'} \vec{V_{Q,c}} \left( (\vec{U_{Q,c}^{h}})^\top \vec{U_{K,c'}^h} \right) (\vec{V_{K,c'}})^{\top}
\end{aligned}
$$

It turns out we can use this equation to understand the QK circuit as a sum of the interactions between pairs of parameter subcomponents. We will use it for a form of static (*data-independent*) and dynamic (*data-dependent*) analysis of the computations of the QK circuit.

We'll need to define two new metrics, one to measure the static interaction strength between pairs of components and another to measure how strongly a pair of subcomponents are interacting on a particular datapoint.

#### QK Circuit - Metric 1: Static Interaction strength

Although we can use <ref>eq:qk-interactions</ref> to understand the *static interaction strength* between subcomponents $c$ and $c'$, we cannot simply use the raw term $\left( (\vec{U_{Q,c}^{h}})^\top \vec{U_{K,c'}^h} \right)$ for a few reasons:

First, because both $\vec{U_c}$ and $\vec{V_c}$ vectors are unnormalized, we need to scale each $\vec{U_c}$ vector by the norm of the corresponding $\vec{V_c}$ vector in order to put the $\vec{U_c}$ vectors on the same scale.

$$
||\vec{V_{Q,c}}|| \left( (\vec{U_{Q,c}^{h}})^\top \vec{U_{K,c'}^h} \right) ||\vec{V_{K,c'}}||
$$

Second, we need to incorporate sequence position information. The above equations actually leave out an important part of our transformer language model: The Rotary Position Embedding (RoPE) rotation matrix <cite>su2024roformer</cite>. For transformers that use RoPE, the QK circuit is actually: $W_{QK, \tau} = (W_Q^{h})^\top \boldsymbol{R}_{\tau} W_K^h$, where $\tau$ is the *offset*—the difference between the sequence position of the query and the key. The rotation matrix rotates the keys and queries by different amounts depending on the offset. Thus we have

$$
\left( ||\vec{V_{Q,c}}||  \vec{U_{Q,c}^{h}} \right)^\top \boldsymbol{R}_{\tau} \left( \vec{U_{K,c'}^h}  ||\vec{V_{K,c'}}|| \right)
$$

Third, and finally, we need to know whether this interaction typically contributes positively or negatively to the attention score. To calculate this, we cheat slightly and import one data-dependent statistic: The sign of the average subcomponent activation for each component on tokens where the subcomponent is causally important. With these three adjustments, we get the Static Interaction Strength:


```equation
tex:
  \htmlClass{hc-ac}{\text{StaticInteractionStrength}(c, c', \tau, h)} 
  \\ =
  \htmlClass{hc-uq}{
    \Big(
      \htmlClass{hc-sign-q}{\text{sign}(\mathbb{E}_x^{(c)} \left[(\vec{V_{Q,c}})^\top x\right])} 
      \cdot 
      \htmlClass{hc-mag-q}{\lVert \vec{V_{Q,c}} \rVert} 
      \cdot 
      \htmlClass{hc-uq-vec}{\vec{U_{Q,c}^h}}
      \Big)^\top
  }
  \htmlClass{hc-r-tau}{ \boldsymbol{R}_{\tau} } 
  \htmlClass{hc-uk}{
    \Big(
      \htmlClass{hc-sign-k}{\text{sign}(\mathbb{E}_x^{(c')} \left[(\vec{V_{K,c'}})^\top x\right])} 
      \cdot 
      \htmlClass{hc-mag-k}{\lVert \vec{V_{K,c'}} \rVert} 
      \cdot 
      \htmlClass{hc-uk-vec}{\vec{U_{K,c'}^h}}
      \Big)
  }  
tips:
  - hc-ac: The static interaction strength between subcomponent c and c' at offset τ in head h
  - hc-uq: The transposed, scaled, signed left-hand vector of subcomponent c in the Q projection matrix of head h
  - hc-sign-q: The sign of the average subcomponent activation of subcomponent c on a dataset of tokens where subcomponent c is causally important
  - hc-mag-q: The magnitude of the right-hand vector of subcomponent c in the Q projection matrix
  - hc-uq-vec: The left-hand vector of subcomponent c in the Q projection matrix of head h
  - hc-r-tau: The RoPE rotation matrix at offset τ
  - hc-uk: The transposed, scaled, signed left-hand vector of subcomponent c' in the Q projection matrix of head h
  - hc-sign-k: The sign of the average subcomponent activation of subcomponent c' on a dataset of tokens where subcomponent c' is causally important
  - hc-mag-k: The magnitude of the right-hand vector of subcomponent c' in the K projection matrix
  - hc-uk-vec: The left-hand vector of subcomponent c' in the K projection matrix of head h
```


The Static Interaction Strength metric is not directly comparable across heads, since each head applies a separate softmax function, making any differences in scales or averages of interaction strength irrelevant. To make the metric comparable across heads, we standardize it:

$$\text{StandardizedStaticInteractionStrength}(c, c', \tau, h) \\ = \frac{\text{StaticInteractionStrength}(c, c', \tau, h) - \mu_h}{\sigma_h}$$

<!-- LaTeX original:
\text{StandardizedAttentionContribution}(c, c', \tau, h)= \frac{W(c, c', h, \tau) - \mu_h}{\sigma_h}
-->

where $\mu_h$ and $\sigma_h$ are the mean and standard deviation of the Static Interaction Strengths across all $(c, c', \tau)$ for head $h$.

For attention layer 1, we plot this metric for each pair of subcomponents for each head and offset (<ref>fig:attn_contrib_grid</ref>). We can see that for some pairs, the Static Interaction Strength changes strongly at different offsets. For these pairs, the same activations might have different effects on the attention at different offsets! For others, the Static Interaction Strengths seem independent of offset, meaning that their effects on the attention scores are determined only by whether data that activate them are present. 

<figure>
<label id="fig:attn_contrib_grid"/>
<img src="figures/layer1_qk_pair_lines_combined.png">
<figcaption>The Standardized Static Interaction Strengths of pairs of parameter subcomponents in the $Q$ and $K$ projection matrices in each head (bottom grid) and all heads (top). The ten pairs with the largest interaction strengths at any offset are shown in color, with the rest in grey. The <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:329</comp> pair exhibit strong postive Static Interaction Strength at early offsets, indicating this pair's involvement in cross-head previous token behavior (and, more generally, 'recent token behavior'.</figcaption>
</figure>

We will use the this plot of Static Interaction Strength to analyze particular attention behaviors. But before we do, we will equip ourselves with a related metric, the Data-Dependent Interaction Strength, which permits dynamic analysis.

#### QK Circuit - Metric 2: Data-Dependent Interaction Strength

The attention patterns of each head depend on how the data interact with the QK circuit: $A^h_\tau = \text{softmax} (x^\top W_{QK, \tau}^{h} x')$. 

We can use <ref>eq:qk-interactions</ref> to decompose the QK circuit and study how the activations $x, x'$ at different timesteps interact with each of the pairs of subcomponents:

$$
\begin{aligned}
Z^h_\tau &= x^\top W_{QK, \tau}^h x' 
&= \sum_{c, c'} x^\top \vec{V_{Q,c}} \left( (\vec{U_{Q,c}^{h}})^\top \boldsymbol{R}_{\tau} \vec{U_{K,c'}^h} \right) (\vec{V_{K,c'}})^{\top} x'
\end{aligned}
$$

Thus, the attention score at each head $h$ and offset $\tau$ consists of the sum of the data's interaction with each of the individual pairs $(c, c')$. On any input, we can therefore decompose the attention score—and hence the attention pattern—into parts that we can study in isolation. This lets us define a data-dependent metric of interaction strength, which forms the basis of our dynamic analysis:

$$
\begin{aligned}
\text{DataDependentInteractionStrength}(c, c', \tau, x, x') 
&= x^\top \vec{V_{Q,c}} \left( (\vec{U_{Q,c}^h})^\top \boldsymbol{R}_{\tau} \vec{U_{K,c'}^h} \right) (\vec{V_{K,c'}})^{\top} x'
\end{aligned}
$$

In <ref>fig:dynamic-1</ref>, you can select which subcomponent interactions to sum together and see the attention score for those pairs. This is a very useful tool, since it splits up any given attention pattern into the contributions of individual, functionally distinct, subcomponent interactions.


```attention
label: fig:dynamic-1
data: data/attention/intro-layer-1.json
```

We'll do an initial analysis of an attention behavior using only these two QK metrics before discussing how they interact with the OV circuit. 

### Decomposing attention behavior 1: Previous token behavior {toc: Behavior 1 - Previous token behavior}


Like many language models, our model has a head that, on average, places the majority of its attention on the previous timestep (<ref>fig:prev_token_scores</ref>). This is typically called a *previous token head* <cite>clark-etal-2019-bert,elhage2021mathematical, olsson2022incontextlearninginductionheads, wang2022interpretability</cite> and, in our model, is head 1 in layer 1 (**L1H1**). However, L1H1 is not the only head to assign substantial probability to the previous token; many other heads do too, including heads in the same layer as L1H1.

<figure class="wide">
<label id="fig:prev_token_scores"/>
<img src="figures/prev_token_scores_combined.png">
<figcaption>Identifying the previous token head: Mean attention across multiple inputs on position $t-1$. 
  **Left**: Average over sequences of random tokens. **Right**: Average over sequences sampled from the dataset. The plots reveal L1H1 is the most canonical "previous token head". But note other heads place substantial average attention on sequence position $t-1$.</figcaption>
</figure> 

Now we need to find subcomponents that might be involved in previous token behavior and establish whether or not their computations span multiple heads. An obvious place to start is by looking at the largest, most frequently active subcomponents in the $W_Q$ and $W_K$ matrices. Perhaps by coincidence, the largest norm subcomponents, <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:329</comp>, are also the most frequently causally important (<ref>fig:qk_comp_weight_norm</ref>)! 

While most subcomponents in layer one are only active on a fraction of tokens, both <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:329</comp> have a CI firing density of $96.7\%$ and $99.8\%$, meaning they're nearly constantly active. Both have the largest weight norm in L1H1, which was the head with the strongest previous token behavior (<ref>fig:qk_comp_weight_norm</ref>). But they also have substantial weight norm in other heads, suggesting they aren't exclusively located in any particular head. Could they be responsible for cross-head previous token behavior?

<ref>fig:attn_contrib_grid</ref> shows that these two components also have very strong offset-dependent Static Interaction Strength. In particular, their interaction is strongest at small offsets, and weak or negative interactions at more distant offsets. This is exactly what we would expect of two components that implement previous token behavior or recent token behavior. This pattern holds not only in L1H1, but also in other heads too. This is strong observational evidence that these two components compute previous token behavior in a way that is distributed across heads.

 

We test this hypothesis causally using ablation and dynamic analysis. When we ablate different $W_Q$ subcomponents on a dataset of prompts, the change in average attention is small for most subcomponent ablations. Only the ablation of <comp key>1.attn.q:316</comp> results in the large reduction of attention at recent offsets (<ref>fig:attn_patterns_q_intv</ref>).

<figure>
<label id="fig:attn_patterns_q_intv"/>
<img src="figures/attn_q_L1_top10_n256_grid.png">
<figcaption>Effect of ablations: Ablating the most active W_Q subcomponents has no distinguishable effect on attention to the recent past except <comp key>1.attn.q:316</comp>, whose ablation reduces attention to small offsets very strongly across all heads that otherwise attended there strongly. Here the baseline is the unablated average attention pattern.</figcaption>
</figure>

<ref>fig:dynamic-1</ref> shows dynamic analysis. For any of the prompts, you can remove the contribution of the <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:329</comp> interaction to the attention score. Removing it destroys the (otherwise usually strong) attention to tokens in the recent past across all heads that had strong to moderate attention there. 

Together, this is strong evidence that the <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:329</comp> interaction computes previous token behavior and is distributed across heads.

This raises a question: What information is this attention moving from the recent past to the current timestep? What *attention values* does this previous token behavior tend to move? Are the different heads carrying forward information from distinct subspaces in the residual stream? Or are they carrying redundant information, perhaps as a form of noise robustness? To study this, we need to analyze the OV circuit, for which we will need another metric. 

#### Previous token behavior employs non-overlapping subspaces in the OV circuit

The OV circuit is made from the $W_V$ and $W_O$ matrices which read from and write to the residual stream respectively.  

$$
W_{OV}^h = W_{O}^h W_{V}^h \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

The attention pattern (which is determined by the QK circuit) determines how 'influential' the OV circuit of previous timesteps is for the output of the attention layer at the current timestep. The vector that gets added to the residual stream is an attention-weighted sum of the outputs of the OV circuits at all previous timesteps: 

$$
\text{AttentionLayer} (x)_t = A^{h \top}_t (W_{OV}^h x_t)  \in \mathbb{R}^{d_{\text{model}}}
$$


Although $W_{OV}^h$ is $d_{\text{model}} \times d_{\text{model}}$ matrix, it only has rank $d_{head}$. Being low rank, each head can therefore only read from and write to a small subspace of the residual stream. It would be useful to know if two heads read from and write to similar subspaces. For this, we will measure the 'overlap' between the subspaces that each head's OV circuit reads from and writes to, for which we'll use the 'Data-weighted Subspace Similarity' metric, which we construct from the Frobenius cosine similarity of the 'read subspaces' and the 'write subspaces' of each head (<ref>fig:prev_tok_ov_overlap_k_329</ref>). See <ref>app:OV-metric-data-frob</ref> for details of how these subspaces are constructed and for further details of this metric. We also measure the Frobenius cosine similarity of the $W_{OV}^h$ matrices themselves (<ref>fig:prev_tok_ov_overlap_k_329</ref>). When calculating similarity, we weight the axes of the read- and write-subspaces by how much data variation lies in each axis, since we do not care as much about weight similarity along axes where data do not exist or do not vary. In all cases, we compare the measured similarities to similarities between random, data-weighted matrices.

Most heads in layer 1, except L1H4, seem at least weakly involved with previous token behavior, as assessed by their previous token score (<ref>fig:prev_token_scores</ref>) and the offset dependence of the Static Interaction Strength of the <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:329</comp> pair (<ref>fig:attn_contrib_grid</ref>). We therefore should look at the overlap in the read and write subspaces of all heads in layer 1 except L1H4. 

The read subspaces of each head are close to or slightly lower than the expected similarity of two random (data-weighted) matrices (<ref>fig:prev_tok_ov_overlap_k_329</ref>). On the other hand, the write subspaces seem close to or slightly higher than the random baseline. These effects seem very weak, but weakly suggest a pattern of attention heads reading from distinct subspaces but writing to slightly less distinct subspaces. 

<figure class="wide">
<label id="fig:prev_tok_ov_overlap_k_329"/>
<img src="figures/layer1_ov_paper_figure_k_329.png">
<figcaption>Data-weighted cosine similarities between each head's $W_{OV}^h$ read and write matrices, and the cosine similarity between each head's raw $W_{OV}^h$. Here, data-weighting uses data where subcomponent <comp key>1.attn.k:329</comp> is causally important. </figcaption>
</figure>

For the head with the strongest previous token behavior, L1H1, the other heads L1H0 and L1H2 seem to read from subspaces with similarities close to the random baseline, but other heads read from much less similar subspaces. When comparing the similarity of the raw $W_{OV}^h$ matrices, there appears to be very little deviation from levels of overlap that would be expected of random matrices, except the comparison between L1H1 and L1H2, which again seem to be more similar than the random baseline. These two heads seem to write to quite different subspaces, though.

Overall, this weakly suggests a picture that previous token behavior spans distinct subspaces across different heads. One potential reason for this is to be able to read more information from the residual stream than might be readable by a single head. There appears to be very limited, but nonzero, redundancy in how heads involved in previous token behavior read from different subspaces, but they largely seem to write to different subspaces. 

<!-- We are excited by the possibilities for understanding attention computations opened up by parameter decomposition.  -->

<!-- A natural question -->

Previous token behavior is an important behavior implemented by probably every language model. But it is far from the only behavior implemented in layer 1. Even in L1H1, only around 60% of attention is on the previous timestep (<ref>fig:prev_token_scores</ref>). What other attention behaviors is this head implementing? In the next section, we look at another behavior implemented by L1H1 in more detail, and examine whether that behavior is also distributed across heads.



### Decomposing attention behavior 2: Previous syntax boundary movement {toc: Behavior 2 - Previous syntactic boundary movement}

Looking again at the static analysis of layer 1, we can see that L1H1 has interactions between Q and K subcomponents that seem to have quite a different offset-dependency (<ref>fig:attn_contrib_grid</ref>). The subcomponents <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:119</comp> seem to interact most strongly at later offsets across multiple heads, including L1H1.

We are already familiar with <comp key>1.attn.q:316</comp>, the query subcomponent that is always active. The key component <comp key>1.attn.k:119</comp> is new: It seems to activate on brackets, punctuation, and newlines, but also some common continuation words, such as 'the' or 'and'. It is causally important on 16% of tokens, which is frequent, but not constantly active. 

This interaction therefore involves a conditional computation: Although <comp key>1.attn.q:316</comp> is always active, constantly looking back in time, the other component <comp key>1.attn.k:119</comp> only interacts with it when it is active. Interestingly, it must be active at an offset that is sufficiently far back in time; otherwise, the Static Interaction Strength may not be strong enough to contribute to the attention score. Almost every head seems to exhibit an offset dependent interaction between subcomponents <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:119</comp>, suggestive of a very distributed computation.

Since this computation is data-dependent, we will benefit from greater use of dynamic analysis. <ref>fig:dynamic-2</ref> shows the attention patterns of all heads. One prompt is shown at a time, but you can select a variety of other prompts. Most importantly, you can select which pairs of Q and K subcomponents should be summed to make the attention scores, and can see their individual Data Dependent Interaction Strength if you select them one at a time.


```attention-multiprompt
label: fig:dynamic-2
data: data/attention/30-dataset-layer-1.json
```

<!-- TODO(Oli)(High priority): These prompts are too short. We should have longer ones. -->

We can see that the interaction between <comp key>1.attn.q:316</comp> and <comp key>1.attn.k:119</comp> contributes significantly to the attention patterns of most heads on previous periods, commas, and newline characters. L1H4 seems capable of maintaining attention on these characters at quite large offsets, whereas other heads seem only to have noticeable attention on them more recently in time. This may be due to competition with other attention score contributions. Prompt X(todo(oli) only once we've changed the dataset examples, todo(oli) above) has an `<|end_of_text|>` token, which activates this pair strongly. 

The activating examples of <comp key>1.attn.k:119</comp> show firings on various forms of punctuation, end of text tokens, newlines, latex "$" symbols, brackets, etc. This suggests that the this pair of subcomponents orchestrates a syntax boundary detector with a variety of short- or long-offset ranges. We'll call this 'previous syntax boundary' movement.

This pair of subcomponents seems responsible for attention to syntax boundary tokens at different ranges in different heads (<ref>fig:attn_contrib_grid</ref>). L1H1 seems to increase self attention upon syntax boundary tokens; L1H2 seems only mildly to attend to syntax boundary tokens and only in the very recent past. L1H5 and L1H0 attends to syntax boundary tokens a small number of tokens in the past. L1H4 seems to attend to syntax boundary tokens many tokens in the past. L1H3 is less clear, but seems to attend to a smaller subset of specific syntax boundary tokens, usually with shorter offset ranges.

The QK circuit of the 'previous syntax boundary movement' behavior seems quite distributed across heads. How does it interact with the OV circuit? We can study this by looking at probability of each key subcomponent being active conditioned on a given value subcomponent being active (<ref>fig:pkv</ref>). The value subcomponents most associated with <comp key>1.attn.k:119</comp> are:

- <comp>1.attn.v:72</comp>
- <comp>1.attn.v:22</comp>
- <comp>1.attn.v:745</comp>
- <comp>1.attn.v:919</comp>
- <comp>1.attn.v:531</comp>
- <comp>1.attn.v:494</comp>
- <comp>1.attn.v:195</comp>
- <comp>1.attn.v:612</comp>
- <comp>1.attn.v:984</comp>
- <comp>1.attn.v:1000</comp>
- <comp>1.attn.v:22</comp>
- <comp>1.attn.v:389</comp>
- <comp>1.attn.v:188</comp>
- <comp>1.attn.v:299</comp>
- <comp>1.attn.v:1014</comp>
- <comp>1.attn.v:227</comp>
- <comp>1.attn.v:946</comp>
- <comp>1.attn.v:340</comp>
- And some with weaker associations (<ref>fig:pkv</ref>).



As in the case of previous token behavior, the data-weighted OV circuits (where we weight the similarity using dataset examples and tokens where <comp key>1.attn.k:119</comp> is causally important) do not seem to read from very similar residual stream subspaces (<ref>fig:prev_tok_ov_overlap_k_119</ref>, though they seem to write to somewhat more similar subspaces than would be expected in random matrices. The OV circuit subcomponents that component <comp key>1.attn.k:119</comp> seems to overlap strongest with are associated with other punctuation and syntax boundary-like tokens across seemingly all heads, in both the read and the write matrices (<ref>app:ov-alignment-k119</ref>). 

To understand why the model is carrying forward information about the previous syntax boundary, we would need to know how the values are being used downstream. But it is easy to surmise at least part of its function: It is useful to know what the previous syntax boundary tokens are in order to perform tasks like closing opened brackets; knowing whether a list is a bullet list or dashed list; or knowing if a token is within or outside of a quotation; and more. 

<!-- TODO(Lee)(Medium priority) potentially insert attn behavior 3: newline pred  -->

Clearly, we have barely scratched the surface of the extent and complexity of attention computations of even this small model. Nonetheless, we are excited by the possibilities for understanding attention computations opened up by decomposing attention layer parameters into parameter components. We believe the breadth of this analysis could be massively increased and note there is significant room for increasing the depth analyzes that use parameter components to decompse and understand attention. We have not, for instance, studied how parameter components could interact across attention layers, perhaps forming structures akin to 'virtual attention heads', but decomposed into their consituent parameter components. 



## Interpreting circuits of parameter components {toc: Circuits}

<label id="sec:circuits"/>

<!-- TODO(Lucius)(Urgent priority): A less abrupt transition/intro to this section. -->


In this section, we use parameter subcomponents to understand at least some aspects of the target model's internal computations on a few different prompts. 

To use parameter decomposition for interpreting multi-step computations, we need a way to study how information flows between parameter components. We do this by calculating attributions, which measure the strength of the interaction between causally important subcomponents on particular prompts. This yields attribution graphs that we can use to study the information flow inside the target model on those prompts. In particular, we use gradient attributions, though we use stop-gradients to measure only the 'direct' effects of one subcomponent on another (<ref>sec:attr-calcs</ref>). 

It should be noted that using gradients in this way 'abstracts away' the complexity of non-linear interactions between subcomponents by summarizing them into a single number. As a result, such attributions are only 'local' measures of interaction strength; their value depends on the particular datapoint that we measure them on. Many works have pointed out issues (such as saturated softmax functions in attention layers) that can cause such local attributions to be unrepresentative of more 'global' measures <cite>kramár2024atpefficientscalablemethod, jafari2025relpfaithfulefficientcircuit</cite>. In order to identify more 'global' measures of interaction strength, we would need to better characterize the nonlinear relationships between parameter subcomponents. This is an important research priority, and one that we've already begun exploring, but not something that this paper covers in detail. We do nonetheless provide analysis that suggests parameter subcomponents of MLP matrices, despite not being directly selected to have simple interactions, tend toward it anyway (<ref>app:interactions-gis-vs-coact</ref>).

<!--Despite those shortcomings, subcomponent attribution graphs do grant us enough insight into the network to tell some interpretable stories about how it calculates its outputs on different prompts.-->


### Attribution calculations

<label id="sec:attr-calcs"/>

To calculate attributions between two subcomponents, we leverage gradients. In particular, we calculate the gradients between each "subcomponent activation", $a^l_c = (\vec{V^l_c})^\top \vec{h^l}$. However, we do not always simply use $\frac{\partial a_{c}}{\partial a_{c'}}$, the partial derivative of the target subcomponent activation $a_{c}$ with respect to the source subcomponent activation. The partial derivative measures the influence of $a_{c'}$ on $a_{c}$ through both *direct* and *indirect* pathways. Understanding the direct effects of a subcomponent give us the clearest mechanistic picture of its role in the network's neural algorithm. We therefore need an attribution method that can distinguish between direct and indirect effects, unlike the partial derivative $\frac{\partial a_{c}}{\partial a_{c'}}$. But, complicating matters further, in models with residual streams a subcomponent's direct effects are not limited only to those in the immediate next layer. The direct effects may skip many layers! 

Instead of using the partial derivative $\frac{\partial a_{c}}{\partial a_{c'}}$, we use the fact that we can control how gradients flow on the backwards pass. We take the partial derivative $\frac{\partial a_{c}}{\partial a_{c'}}$, but we stop the gradients flowing through all subcomponents that are not the source subcomponent (<ref>fig:attr-graph-expl</ref>). This avoids measuring their effects on the target node, including the indirect effects of the source node that flow through them.

<figure class="fig-attr-graph-expl">
<label id="fig:attr-graph-expl"/>
<img src="figures/Explaining attribution graphs.png">
<figcaption>To exclude indirect effects (i.e. effects that one parameter subcomponent has on another that are mediated by intermediate parameter subcomponents), we stop the gradients flowing through all subcomponents that are not the source subcomponent.</figcaption>
</figure>

This derivative approximates how sensitive the target node is to the source node. Our attribution multiplies this "sensitivity" by the strength of the activation of the source node in order to measure its overall influence. Additionally, we do not want to include causally unimportant nodes in our attributions, and therefore multiply the resulting term by the source subcomponent's causal importance:

<!-- $\text{attr}(s \to t) = \sum_{\text{batch}} \sum_{\text{pos}} \frac{\partial a_t}{\partial a_s} \cdot a_s \cdot \text{CI}(s, \text{pos})$ -->

$$\text{attr}(c' \to c) = \left( \frac{\partial a_c}{\partial a_{c'}} \right)^* \cdot a_{c'} \cdot g_{c'}$$
<!-- LaTeX original:
\text{attr}(s \to t) = \left( \frac{\partial a_t}{\partial a_s} \right)^* \cdot a_s \cdot g_s
-->

where the $*$ around the partial derivative denotes stopped gradients on non-source subcomponents.
<!-- TODO(Dan)(Low priority): Maybe the figure should use this ()* notation, and then define attr(A->D) later. It's slightly weird in the figure that we have the attr(A->D) but we don't say anything about the CIs or multiplying by the source subcomponent activation  -->

For more details on our gradient attributions, see <ref>app:gradient_attributions</ref>.







### Pruning for specific behaviors

<label id="sec:attr-graph-post-proc"/>


Most prompts, even simple ones, tend to activate hundreds of parameter subcomponents, which is too many to analyze at once! 

We can further reduce the number of subcomponents we need to analyze by keeping only those subcomponents involved in computing some particular output behavior on a prompt that we are interested in. 

For example, on the prompt `The` `·princess` `·lost` `·her` `·crown` `.`, suppose we wanted to analyze how the model successfully predicts `·her`. We would therefore only be interested in subcomponents that were involved in computing this specific predicition at this specific sequence position, which is a smaller subset of subcomponents than the set used to predict all tokens at all sequence positions. We can therefore find new causal importances to identify only those subcomponents. We can optimize new causal importances using a cross-entropy reconstruction loss on the label `·her` on the sequence position for `·lost`, instead of a KL-divergence to all the target model's output probabilities on all sequence positions of the prompt.

As in VPD base training we optimise causal importances under both stochastic and adversarial mask sampling to ensure the resulting graphs are mechanistically faithful. For details about this technique, see <ref>app:posthoc_ci</ref>. 

One might wonder whether adversarial sampling is actually necessary for mechanistic faithfulness for this post-hoc pruning. After all, the parameter subcomponents are now frozen and only the causal importances can change, so the optimisation has much fewer degrees of freedom to create spurious graphs that score well on the loss. 

To investigate this, we also create graphs optimised without stochastic or adversarial sampling in the two case studies below, using just the causal importances as masks. We compare those graphs to graphs pruned with adversarial masking. As we will see, the non-adversarially pruned graphs often look interpretable. However, they contain far fewer subcomponents than the adversarially pruned graphs and fail to capture many aspects of the target model's computations, making them look much simpler and more superificial than they actually are. Further confirming the mechanistic unfaithfulness of the non-adversarially pruned graphs, they often score much better on the task than the actual target model, reaching near $100\%$ accuracy.<footnote>We also found that graphs pruned with stochastic sampling but no adversarial sampling often seemed to be very mechanistically unfaithful as well, but we do not show these results here.</footnote> 

We believe that this problem seem likely to generalise to any setting in which pruning is used to identify subsets of nodes in large causal graphs important for downstream tasks.



<!--We can apply this method for other behaviors of interest in order to identify only the subcomponents involved in those behaviors. We will use this approach in the case studies below.-->
<!-- This allows us to further reduce the number of components we need to analyze to understand some behavior of the target model.  -->






<!--## Case studies: Interpretability in language model parameter space-->


<!--
+┌────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐   
+│                 Syntax                 │                                       Effect                                        │ 
+├────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
+│ <comp>3.attn.o:2:281</comp>            │ Default. Shows the component's autointerp label as text (formatted key as tooltip). │   
+│                                        │  Falls back to formatted key if no label exists.                                    │   
+├────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤   
+│ <comp key>3.attn.o:2:281</comp>        │ Forces display of the formatted key (e.g. L3.Attn.O.281) instead of the autointerp  │
+│                                        │ label. Autointerp label becomes the tooltip.                                        │   
+├────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
+│ <comp hidden>output:2:617</comp>       │ Renders with display:none. Lets you highlight nodes in the graph (like output       │   
+│                                        │ nodes) without rendering any visible text for them.                                 │
+├────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
+│ <comp                                  │ Prevents graph-node highlighting on hover. Useful for mentioning a component        │
+│ no-highlight>3.attn.o:2:281</comp>     │ without drawing visual attention to the graph.                                      │   
+├────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
+│ <comp key                              │ Combines attributes: shows the formatted key and suppresses graph highlighting.     │   
+│ no-highlight>1.attn.q:316</comp>       │                                                                                     │   
+└────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘
+-->

### Case study 1: Gendered possessive pronoun

<label id="sec:case-studies-pronoun"/>

On the prompt `The` `·princess` `·lost` `·her` `·crown` `.` the target model correctly predicts with high probability ($0.586$) that `·her` follows `·lost`. This requires recognizing that a possessive pronoun is likely to come next, remembering that the previous token was `·princess`, and knowing that princesses are predominantly associated with female pronouns. How does the model perform this task? 

We can use attribution graphs to follow the flow of information between parameter components and see what information is processed and by which parameters.

<label id="graph:princess"/>
```graph
id: princess-full
data: data/graphs/princess-full.json
details: data/graphs/princess-full-details.json
caption: Attribution graph for predicting `·her` on the prompt "The princess lost her crown.", pruned with adversarial sampling.<footnote>Coefficient $0.5$ for cross-entropy reconstruction with stochastic sampling, coefficient $0.5$ for cross entropy with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.09$, $p=0.3$, $2000$ optimization steps.</footnote> There are 150 subcomponents in the graph. 
```

<graph-explanation name="princess-full">
  
Attribution graph for the prompt `The` `·princess` `·lost` `·her` `·crown` `.` after adversarial pruning, keeping only the subcomponents that matter for predicting the output `·her` after `·lost`.<footnote>Using $2000$ optimization steps, cross-entropy reconstruction with stochastic sampling, loss coefficient $0.5$, cross entropy with $4$ steps of PGD at lr$=1$, coefficient $0.5$, and importance minimality loss coefficient $0.09$, $p=0.3$.</footnote> The graph has a total of 150 subcomponents. The target model assigns probability $0.586$ to the output `·her`. Causal importance masking with the nodes in this graph increases that probability to $1.000$ and stochastic masking increases it to $0.999$. However, adversarial masking decreases the probability on the output `·her` to $0.443$, which indicated that this graph still isn't quite capturing all the relevant computation going on in the model.

Working backward from the output, we will see that the top two positive attributions to the output node `·her` in the graph come from two different computational pathways.


<graph-page-break/>
  
<graph-comp hidden>output:2:617</graph-comp>

**Pathway 1** 

This pathway appears to carry information about the 'femaleness' of the `·princess` token forward in time to make the pronoun prediction `·her`. Working backward from output node to input nodes:

The largest positive attributions to the output token `·her` is from a layer 3 attention output subcomponent labeled <graph-comp index>3.attn.o:2:281</graph-comp>. Ablating it out of the target model changes the top prediction to `·his`. That subcomponent, in turn, receives its largest attribution edges from a subcomponent of the attention layer 3's $W_K$ at the `·princess` sequence position, which is causally important on almost every token (<graph-comp key>3.attn.k:1:145</graph-comp>), and a component of the attention layer 3's $W_V$, likewise on the `·princess` sequence position, labeled <graph-comp>3.attn.v:1:676</graph-comp>. 

The $W_V$ component in turn receives its top attribution from <graph-comp key>0.mlp.down:1:3473</graph-comp>, which appears to be polysemantic. It is active on various female names and other words and sentences associated with or about women, but also on in a range of other contexts, perhaps particularly scientific ones. Its top attribution comes from a component of the layer 0 MLP up projection matrix labeled <graph-comp>0.mlp.up:1:327</graph-comp>, which then connects straight to the `·princess` input embedding. 

In summary, this pathway appears to carry a femaleness attribute from the `·princess` sequence position to the `·lost` sequence position using the layer 3 attention. The relevant key and query subcomponents almost always fire, indicating that this attention routing happens as part of the generic previous token behavior.

<graph-page-break/>

<graph-comp hidden>output:2:617</graph-comp>

**Pathway 2**

The second largest positive attributions to the output `·her` is from the layer 2 MLP down projection subcomponent <graph-comp key>2.mlp.down:2:773</graph-comp>. It seems to also be causally important when the model is about to predict an object pronoun, among other things (though this detail seems to have been missed by its autointerp label <graph-comp>2.mlp.down:2:773</graph-comp>). 

The strongest attribution to this subcomponent, in turn, comes from a layer 2 MLP input subcomponent labeled <graph-comp>2.mlp.up:2:401</graph-comp><footnote>It indeed appears to be causally important primarily on tokens that are verbs. Notably, whether a token is classified as a verb for this purpose is context-dependent. For example, in the sentence `I'd` `·like` `·to` `·do` `·something` `·like` `·this`, the subcomponent has high activation ($12.7$, $19.1$) and causal importance $1.0$ on `do` and the first `like` token, but low activation ($2.9$) and causal importance $0$ on the second `like` token.</footnote>. 

This component receives attribution from a diverse set of verb-related layer 0 MLP subcomponents, such as <graph-comp no-highlight>0.mlp.up:2:3063</graph-comp> and  <graph-comp no-highlight>0.mlp.down:2:1189</graph-comp>, which then connect to the `·lost` embedding.

In summary, this pathway appears to upweight object pronoun predictions based on detecting the verb `·lost` in the input.

</graph-explanation>

The top two pathways in the adversarially pruned graph suggest two core mechanisms: one which moves the femaleness attribute of `·princess` over to the next token via attention layer 3, and another which detects the verb `·lost` via MLP layer 2 and suggests that an object pronoun might follow.

If we prune the graph for high probability on `·her` using only the causal importances as masks, neglecting adversarial robustness, we recover a graph of just six components (<ref>graph:princess_ci_masked</ref>), which corresponds almost exactly to the most attributed subcomponents in these same two top pathways.


<label id="graph:princess_ci_masked"/>
```graph
id: princess-minimal
data: data/graphs/princess-minimal.json
details: data/graphs/princess-minimal-details.json
caption: Attribution graph for predicting `·her` on the prompt "The princess lost her crown.", pruned with causal importance masking.<footnote>Coefficient $1.0$ for cross-entropy reconstruction with causal importance masking, importance minimality coefficient $1.0$, $p=0.3$, $2000$ optimization steps.</footnote>
```

<graph-explanation name="princess-minimal">
  
<graph-comp hidden>output:2:617</graph-comp>
Attribution graph for predicting `·her` on the prompt `The` `·princess` `·lost` `·her` `·crown` `.`, pruned with causal importance masking.<footnote>Using a cross-entropy reconstruction with causal importance masking, importance minimality coefficient $1.0$, $p=0.3$, $2000$ optimization steps.</footnote> There are 6 subcomponents in total, forming two distinct pathways. The subcomponents in these two pathways correspond almost exactly to the most strongly attributed subcomponents in the two top pathways of the much larger adversarially pruned graph depicted in <ref>graph:princess</ref>. 

**Pathway 1:** From the `·princess` embedding to components in the layer 0 MLP up and down projection matrices labeled <graph-comp>0.mlp.up:1:327</graph-comp> and <graph-comp>0.mlp.down:1:3473</graph-comp>, to a component in the layer 3 attention value matrix labeled <graph-comp>3.attn.v:1:676</graph-comp> to a subcomponent in the layer 3 attention output matrix on the `·lost` sequence position labeled <graph-comp>3.attn.o:2:281</graph-comp>. 

**Pathway 2:** From the `·lost` emebedding to subcomponents in the layer 2 MLP up and down projection matrices labeled <graph-comp>2.mlp.up:2:401</graph-comp> and <graph-comp>2.mlp.down:2:773</graph-comp>.

For more discussion of the subcomponents in these pathways, see <ref>graph:princess</ref>.

The target model assigns probability $0.586$ to the output `·her`. Causal importance masking with the six subcomponents in this small graph increases this probability to $0.895$, and stochastic masking based on the causal importances in this graph increases it even more, up to $0.969$. 

One might then falsely suppose that these six subcomponents perform all the important computation for this pronoun prediction task and the other subcomponents are, if anything, just a hindrance. But evaluation with adversarial masking based on the causal importances in this graph<footnote> With $4$ PGD optimization steps at learning rate $1$.</footnote> drops the probability on the `·her` prediction down to $<0.0005$, revealing that this isn't true at all.
</graph-explanation>

This confirms that these six subcomponents are sufficient for reproducing the desired output. This much smaller graph even generalises to slightly different prompts: On the input `The` ` lady` `·lost` `·her` `·crown` `.`, a forward pass using only the six subcomponents in the small graph at the exact same sequence positions also recovers the target model's `·her` prediction<footnote>With output probability $0.895$ under causal importance masking, and $0.275$ under stochastic masking.</footnote>.
But the lack of adversarial robustness in the smaller graph confirms that it does not provide anything close to a full account of the relevant computation going into the model's prediction.<footnote>Pruning with stochastic masking doesn't perform any better. A graph for the princess prompt we pruned with stochastic masking ended up with $14$ subcomponents in total, and still assigned probability $<0.0005$ under adversarial masking.</footnote> All 150 components in <ref>graph:princess</ref> likely play some role — otherwise the optimization would have pruned them. While these six components suffice to put high probability on `·her`, they fail to suppress other computational pathways that would predict different outputs. We do not attempt to fully understand the complete graph here. 

<!--<comp hidden>output:2:617</comp>
The largest negative attribution to `·her` comes from <comp>3.mlp.down:2:3217</comp>. This component is causally important whenever the model predicts a pronoun, but its role varies: it occurs with both positive activation (increasing pronoun probability) and negative activation (as on this prompt), where ablating it decreases pronoun probability.-->

<!-- TODO(Lucius)(High priority): TO DISCUSS Before launching into a new prince prompt, we need to motivate why we need to look at a new prompt at all. Lucius: Better now?-->

We also investigate the symmetrical case for male pronouns on the prompt `The` `·prince` `·lost` `·his` `·crown` `.`, where the target model predicts `·his` with probability $0.512$, and similar results.

<label id="graph:prince-full"/>
```graph
id: prince-full
data: data/graphs/prince-full.json
details: data/graphs/prince-full-details.json
caption: Attribution graph for predicting `·his` on the prompt "The prince lost his crown.", pruned with adversarial sampling.<footnote>Coefficient $0.5$ for cross-entropy reconstruction with stochastic sampling, coefficient $0.5$ for cross entropy with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.05$, $p=0.3$, $2000$ optimization steps.</footnote> There are 160 subcomponents in the graph. The target model assigns probability $0.512$ to `·his`.
```

<graph-explanation name="prince-full" >
Attribution graph for predicting `·his` on the prompt `The` `·prince` `·lost` `·his` `·crown` `.`, pruned with adversarial sampling.<footnote>Here we use coefficient $0.5$ for the cross-entropy reconstruction loss with stochastic sampling, coefficient $0.5$ for cross entropy loss with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.05$, $p=0.3$, $2000$ optimization steps.</footnote> The graph has a total of 160 subcomponents. The target model assigns probability $0.512$ to `·his`. Causal importance masking with the nodes in this graph increases that probability to $1.000$ and stochastic masking increases it to $0.998$. However, adversarial masking<footnote>With 4 PGD optimization steps at learning rate $1$.</footnote> decreases the probability on `·his` to $0.383$, which indicates that this graph still isn't quite capturing all the relevant computation going on in the model.

The graph is structurally similar to the adversarially pruned graph for the `·princess` prompt in <ref>graph:princess</ref>. 95 of the 150 subcomponents in that graph also show up at the same sequence position in this graph, including the subcomponents we discussed that form a pathway for upweighting object pronoun predictions based on detecting the verb `·lost` in the input. However, as we might expect, the subcomponents for moving the femaleness attribute to the next sequence position is not present here.
<!-- Over half the components are shared between the two graphs, including  <comp no-highlight>2.mlp.down:2:773</comp> in the MLP layer 2 down projection,  <comp no-highlight>3.mlp.down:2:3217</comp> in the MLP layer 3 down projection, and  <comp no-highlight>3.mlp.down:2:3498</comp> in the MLP layer 3 down projection.-->

</graph-explanation>


As with the princess prompt, pruning with CI masking instead of adversarial masking recovers a much smaller graph of just six subcomponents organised into two pathways that is sufficient to compute the `·his` prediction, but isn't adversarially robust at all. 

<label id="graph:prince-minimal"/>
  
```graph
id: prince-minimal
data: data/graphs/prince-minimal.json
details: data/graphs/prince-minimal-details.json
caption: Attribution graph for predicting `·his` on the prompt "The prince lost his crown.", pruned with causal importance masking.
```

<graph-explanation name="prince-minimal" >
Attribution graph for predicting `·his` on the prompt `The` `·prince` `·lost` `·his` `·crown` `.`, pruned with causal importance masking.<footnote>coefficient $1.0$ for cross-entropy reconstruction with causal importance masking, importance minimality coefficient $1.0$, $p=0.3$, $2000$ optimization steps.</footnote> 

The six subcomponents in this graph form two pathways, mirroring the two pathways in the graph for the princess prompt. 

**Pathway 1**: From the `·prince` embedding to subcomponents in the layer 0 MLP up and down projection matrices labeled <graph-comp no-highlight>0.mlp.up:1:2822</graph-comp> and <graph-comp no-highlight>0.mlp.down:1:3455</graph-comp>, to components in the layer 3 attention value and output matrices labeled <graph-comp no-highlight>3.attn.v:1:1010</graph-comp> and  <graph-comp no-highlight>3.attn.o:2:776</graph-comp>.

Compared to the four subcomponents in the corresponding core pathway moving the female attribute from `·princess` to `·lost` in <ref>graph:princess</ref>, these four subcomponents seem less gender specific, firing in both male and female contexts, though more often male ones. This suggests a mechanism under which male pronoun prediction is the default unless actively contradicted. Reinforcing this hypothesis, running the princess prompt with just the six subcomponents in this graph results in the model predicting  `·his` rather than `·her`.

**Pathway 2**: From the `·lost` embedding to a component in the layer 0 MLP up projection matrix labeled <graph-comp no-highlight>0.mlp.up:2:3063</graph-comp>, to a subcomponent in the layer 0 MLP down projection matrix labeled <graph-comp no-highlight>0.mlp.down:2:1189</graph-comp>. These two subcomponents also formed part of the  second core pathway for the `·princess` prompt we discussed before, see <ref>graph:princess</ref>. 

</graph-explanation>

We stress again that the above is far from a complete account of the meaningful computation going on in the model for these input prompts. We have merely traced out the flow of information between a subset of components that are sufficient for computing the output, which is much smaller than the subset of components that are actually involved in computing the output. 


### Case study 2: Bracket closing

<label id="sec:case-studies-bracket"/>

On the prompt `<` `u` `,` `v` `>` the target model correctly predicts that `>` follows `v`, assigning probability $0.547$. This requires the model to remember that, earlier in the sentence, `<` opened a bracket that now needs to be closed. How does the model perform this task?

<label id="graph:bracket"/>

```graph
id: bracket-full
data: data/graphs/bracket-full.json
details: data/graphs/bracket-full-details.json
caption: Attribution graph for predicting `>` after `v` on the prompt `<` `u` `,` `v` `>`, pruned with adversarial sampling.
```


<graph-explanation name="bracket-full">
Attribution graph for predicting `>` after `v` on the prompt `<` `u` `,` `v` `>`, pruned with adversarial sampling.<footnote>Coefficient $0.5$ for cross-entropy reconstruction with stochastic sampling, coefficient $0.5$ for cross entropy with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.05$, $p=0.3$, $4000$ optimization steps.</footnote> The target model predicts `>` after `v` with probability $0.547$. 

Most of the 158 subcomponents in the graph appear to be specialised for predicting closing delimiters, closing angled brackets more specifically, or closing angled brackets in particular, spanning large subspaces within the model. 

<graph-page-break/>
  
<graph-comp hidden>output:3:31</graph-comp>
The two largest positive attributions to the output `>` come from:

1. A layer 3 MLP down projection matrix subcomponent labeled <graph-comp>3.mlp.down:3:1414</graph-comp>

2. A layer 2 MLP down projection matrix subcomponent labeled <graph-comp>2.mlp.down:3:1560</graph-comp>

Ablating these two subcomponents out of the target model severely degrades the `>` prediction, lowering the probability from $0.547$ to $0.158$ and $0.243$ for individual ablations, and to $0.046$ for joint ablation. The model instead reassigns probability mass to other delimiters such as `)`, `_`, `,` or `)$`, suggesting that the model still knows there is an open delimiter to close, but not that it is a right angled bracket in particular.

These subcomponents must rely on information about the open angled bracket received from the previous sequence position. We can see in the graph that information is carried from the `<` position to the `v` position through attention at layers 1, 2, and 3. In the following, we will give a brief survey of the attention subcomponents involved in this transfer.

<graph-page-break/>
  
**Layer 1 attention summary**
<graph-comp hidden>1.attn.q:3:316</graph-comp>
<graph-comp hidden>1.attn.k:0:119</graph-comp>
<graph-comp hidden>1.attn.k:0:329</graph-comp>
<graph-comp hidden>1.attn.k:2:329</graph-comp>
<graph-comp hidden>1.attn.v:0:22</graph-comp>
<graph-comp hidden>1.attn.v:0:984</graph-comp>
<graph-comp hidden>1.attn.v:0:249</graph-comp>
<graph-comp hidden>1.attn.v:0:788</graph-comp>
<graph-comp hidden>1.attn.v:0:102</graph-comp>
<graph-comp hidden>1.attn.v:0:474</graph-comp>
<graph-comp hidden>1.attn.v:0:504</graph-comp>
<graph-comp hidden>1.attn.v:0:571</graph-comp>
<graph-comp hidden>1.attn.v:2:22</graph-comp>
<graph-comp hidden>1.attn.v:2:984</graph-comp>
<graph-comp hidden>1.attn.v:2:299</graph-comp>
<graph-comp hidden>1.attn.v:1:428</graph-comp>
<graph-comp hidden>1.attn.o:3:899</graph-comp>
<graph-comp hidden>1.attn.o:3:91</graph-comp>
<graph-comp hidden>1.attn.o:3:300</graph-comp>
<graph-comp hidden>1.attn.o:3:187</graph-comp>
<graph-comp hidden>1.attn.o:3:362</graph-comp>


As we will see in the following, interpretations of the query and key subcomponents in layer 1 suggest that information about the preceding open angled bracket is moved from the `<` sequence position to the `v` sequence position in this layer as a result of both generic previous token behavior, and as part of a behavior that moves information at formatting boundaries to following sequence positions. 

Ablating the layer 1 attention output components out of the target model on the `v` sequence position degrades performance severely, with the model now assigning just $0.015$ probability to `>` instead of $0.547$. Its top logit instead becomes `<|endoftext|>`, with probability $0.056$.

Similarly, ablating the layer 1 attention output components out of the graph reduces the probability the adversarialy masked forward pass puts on `>` down to $0.021$. However, with causal importance masking, the probability assigned to `>` stays at $\approx 1.000$. This once again indicates that using naive masking schemes to infer causality can be very misleading, and adversarial sampling can help us avoid underestimating the number of components involved in the target model’s computation.

<graph-page-break/>
  

**Layer 1 attention query and key matrices**

- A single query subcomponent labeled <graph-comp>1.attn.q:3:316</graph-comp> on the `v` sequence position. This indicates that the relevant query at this layer is triggered as part of the generic previous token behavior. 
- Two key subcomponents on the `<` sequence position, labled <graph-comp>1.attn.k:0:329</graph-comp> and <graph-comp>1.attn.k:0:119</graph-comp>. This indicates that the `<` sequence position is attended to in this layer partially as part of generic previous token behavior, and partially as part of a behavior that moves information at formatting boundaries.
- The key subcomponent labeled <graph-comp>1.attn.k:2:329</graph-comp> is also kept on the `,` sequence position, indicating that the relevant information there is attended to purely as part of generic previous token behavior.

<graph-page-break/>

**Layer 1 attention value matrix** 
<graph-comp hidden>1.attn.v:0:504</graph-comp>
<graph-comp hidden>1.attn.v:0:571</graph-comp>
<graph-comp hidden>1.attn.v:2:299</graph-comp>
<graph-comp hidden>1.attn.v:1:428</graph-comp>
<graph-comp hidden>1.attn.v:2:22</graph-comp>
<graph-comp hidden>1.attn.v:2:984</graph-comp>

There are eight value subcomponents on the `<` sequence position.

- Two components labeled <graph-comp>1.attn.v:0:22</graph-comp> and <graph-comp>1.attn.v:0:984</graph-comp>, which also appear on the `,` position, as one might expect since they seem related to a wider set of delimiter syntax that also includes commas.
- Three subcomponents labeled <graph-comp>1.attn.v:0:249</graph-comp>, <graph-comp>1.attn.v:0:788</graph-comp> and <graph-comp>1.attn.v:0:102</graph-comp>, which appear more specialised to angled brackets in particular. Their activations and causal importances also tend to be much lower for closing angled brackets than opening angled brackets. <graph-comp>1.attn.v:0:102</graph-comp> is also part of a larger component that also has two subcomponents in the layer 2 attention value matrix of this graph. Subcomponents in this component all seem to be causally important primarily on various left angle brackets, like `<`, `></ `, `} <` etc.
- One subcomponent labeled <graph-comp>1.attn.v:0:474</graph-comp> fires on angled brackets, again more strongly for opening angled brackets, but also a few other delimiter types, such as `:` after `A` in the context of a Q&A.

<graph-page-break/>

**Layer 1 attention value matrix** (*continued*)
<graph-comp hidden>1.attn.v:0:22</graph-comp>
<graph-comp hidden>1.attn.v:0:984</graph-comp>
<graph-comp hidden>1.attn.v:0:249</graph-comp>
<graph-comp hidden>1.attn.v:0:788</graph-comp>
<graph-comp hidden>1.attn.v:0:102</graph-comp>
<graph-comp hidden>1.attn.v:0:474</graph-comp>
<graph-comp hidden>1.attn.v:0:504</graph-comp>
<graph-comp hidden>1.attn.v:0:571</graph-comp>
<graph-comp hidden>1.attn.v:2:22</graph-comp>
<graph-comp hidden>1.attn.v:2:984</graph-comp>
<graph-comp hidden>1.attn.v:2:299</graph-comp>
<graph-comp hidden>1.attn.v:1:428</graph-comp>

On the `<` sequence position:

- One component labeled <graph-comp>1.attn.v:0:504</graph-comp>, which fires on opening brackets more generally, including e.g. `{`, `[`, and variations like `\^ {`, as well as some delimiters like `;`, though apparently only in technical and math heavy contexts, and a few closing brackets like `);`, `}`. Again, the subcomponents’ activation on these closing brackets is notably lower than on the opening brackets.
- Finally, one subcomponent labeled <graph-comp>1.attn.v:0:571</graph-comp>, which is active almost exclusively on the first or first few tokens in a sequence.

On the other two sequence positions:


- There are three subcomponents on the `,` sequence position. Two also appear on the `<` sequence position, see previous page. The third is labeled <graph-comp>1.attn.v:2:299</graph-comp>.
- A subcomponent labeled <graph-comp>1.attn.v:1:428</graph-comp> is the subcomponent in the layer 1 attention on the `u` sequence position.

<graph-page-break/>
  
**Layer 1 attention output matrix**

There are five subcomponents in the layer 1 attention output matrix on the `v` sequence position:

- One component, labeled <graph-comp>1.attn.o:3:899</graph-comp>, appears to be active primarily whenever an open left angled bracket (`<`, `.<`, etc.) has not been closed yet, or when the previous token was a backslash (`\`, `$\ `, etc.).
- Another component, labeled <graph-comp>1.attn.o:3:91</graph-comp>, seems to be active on and everywhere between separators and delimiters like commas or semicolons in lists, and various brackets in math or code.
- A subcomponent labeled <graph-comp>1.attn.o:3:300</graph-comp>, which seems to likewise activate primarily on tokens between delimeters, in this case seemingly exclusively various kinds of brackets in latex or code.
- One component, labeled  <graph-comp>1.attn.o:3:187</graph-comp> appears to be active on any markup, HTML or other code and, seemingly to a somewhat lesser extent, on latex.
- The final component, labeled  <graph-comp>1.attn.o:3:362</graph-comp>, was somewhat difficult for us to make sense of. It fires on short text passages in succession, as if it is predicting something from the moment some left delimeter is seen until some other right delimiter is hit, but we could not determine from the examples what those delimiters are.

<graph-page-break/>

**Layer 2 attention summary** 
<graph-comp hidden>2.attn.q:3:270</graph-comp>
<graph-comp hidden>2.attn.q:3:279</graph-comp>
<graph-comp hidden>2.attn.k:0:197</graph-comp>
<graph-comp hidden>2.attn.k:0:347</graph-comp>
<graph-comp hidden>2.attn.k:0:204</graph-comp>
<graph-comp hidden>2.attn.k:0:206</graph-comp>
<graph-comp hidden>2.attn.v:0:121</graph-comp>
<graph-comp hidden>2.attn.v:0:484</graph-comp>
<graph-comp hidden>2.attn.v:0:234</graph-comp>
<graph-comp hidden>2.attn.v:0:961</graph-comp>
<graph-comp hidden>2.attn.v:0:22</graph-comp>
<graph-comp hidden>2.attn.v:0:65</graph-comp>
<graph-comp hidden>2.attn.v:0:473</graph-comp>
<graph-comp hidden>2.attn.v:0:394</graph-comp>
<graph-comp hidden>2.attn.v:0:927</graph-comp>
<graph-comp hidden>2.attn.o:3:161</graph-comp>
<graph-comp hidden>2.attn.o:3:433</graph-comp>
<graph-comp hidden>2.attn.o:3:963</graph-comp>
<graph-comp hidden>2.attn.o:3:855</graph-comp>
<graph-comp hidden>2.attn.o:3:359</graph-comp>
<graph-comp hidden>2.attn.o:3:722</graph-comp>
<graph-comp hidden>2.attn.o:3:878</graph-comp>
<graph-comp hidden>2.attn.o:3:218</graph-comp>
<graph-comp hidden>2.attn.o:3:529</graph-comp>
<graph-comp hidden>2.attn.o:3:1000</graph-comp>
<graph-comp hidden>2.attn.o:3:286</graph-comp>
<graph-comp hidden>2.attn.o:3:495</graph-comp>
<graph-comp hidden>2.attn.o:3:121</graph-comp>
<graph-comp hidden>2.attn.o:3:735</graph-comp>

Judging by the attribution lines in the graph, layer 2 seems to attend to information at the `<` sequence position from the `v` sequence position in part because the information received at the previous attention layer triggering a more closing-delimiter specific query that searches for a preceding opening-delimiter key. So, the two layers do not just operate in parallel, they also at least partially compose in series.

Just as with layer 1, ablating the layer 2 attention output subcomponents on the `v` position out of the target model severely degrades performance. The model then still expects some kind of bracket, but not an angled bracket in particular. For example, the probability it assigns to `)` increases from $0.079$ to $0.279$, the probability it assigns to ` ] ` increases from $0.015$ to $0.075$, and the probability it assigns to `);` increases from $0.004$ to $0.052$. The probability it assigns to `>` decreases from $0.547$ to $0.02$. This indicates that the information carried by the value and output subcomponents in this attention layer is important for distinguishing which specific kind of left bracket needs to be closed with sufficient confidence.


<graph-page-break/>
  
**Layer 2 attention query and key matrices**

- There are two query components on the `v` sequence position. The first is labeled <graph-comp>2.attn.q:3:270</graph-comp>, the second <graph-comp>2.attn.q:3:279</graph-comp>. They receive high positive attribution from both the layer 0 MLP down projection components and the layer 1 attention output components. Specifically, the latter subcomponent receives high positive attribution from the layer 1 attention output component <graph-comp key no-highlight>1.attn.o:3:187</graph-comp> and a little from <graph-comp key no-highlight>1.attn.o:3:899</graph-comp>. This suggests that this query is partially triggered by the received closed angled bracket information from the layer 1 attention, as part of a compositional pathway inolving two attention layers in series.
- There are four key matrix subcomponents on the `<` sequence position.
 Two, labeled <graph-comp>2.attn.k:0:197</graph-comp> and <graph-comp>2.attn.k:0:347</graph-comp> fire on various opening brackets such as `<`, `(` and `[`, as well as other delimiters like opening quotation marks, `$` in latex, `**` and variations of these created by the tokeniser, like `[@`, `(*`, `_{`, `![ ` and such. The third is labeled <graph-comp>2.attn.k:0:204</graph-comp>, and the final one <graph-comp>2.attn.k:0:206</graph-comp>.


<graph-page-break/>
  
**Layer 2 attention value matrix**

There are nine value components on the `<` sequence position:

- Two, labeled <graph-comp>2.attn.v:0:121</graph-comp> and <graph-comp>2.attn.v:0:484</graph-comp> are part of the same "left angled brackets" cluster that also had a subcomponent in the layer 1 attention value matrix of the graph at this same sequence position.
- Another two, labeled <graph-comp>2.attn.v:0:234</graph-comp> and <graph-comp>2.attn.v:0:961</graph-comp> are part of another cluster of four components that seem to fire on left angled braces, but also left curly braces, opening quotation markers, and the start of links.
- The other five subcomponents are labeled <graph-comp>2.attn.v:0:22</graph-comp>, <graph-comp>2.attn.v:0:65</graph-comp>, <graph-comp>2.attn.v:0:473</graph-comp>, <graph-comp>2.attn.v:0:394</graph-comp> and <graph-comp>2.attn.v:0:927</graph-comp>, and likewise variously fire on left angled brackets, left brackets in general, left delimiters somewhat more generally, and in one case both left and right delimiters. Some of them are also causally important on the tokens after left delimiters as well, as though they are responding to the delimiters information being carried forward from the previous sequence position.

<graph-page-break/>

**Layer 2 attention output matrix** 

There are fourteen attention output subcomponents on the `v` sequence position:

- Two, labeled <graph-comp>2.attn.o:3:161</graph-comp> and <graph-comp>2.attn.o:3:433</graph-comp> seem to fire whenever there are unclosed left delimiters, particularly left angled brackets, but left round, curly or boxy brackets.
- Eight subcomponents, labeled (<graph-comp>2.attn.o:3:963</graph-comp>, <graph-comp>2.attn.o:3:855</graph-comp>, <graph-comp>2.attn.o:3:359</graph-comp>, <graph-comp>2.attn.o:3:722</graph-comp>, <graph-comp>2.attn.o:3:878</graph-comp>, <graph-comp>2.attn.o:3:218</graph-comp>, <graph-comp>2.attn.o:3:529</graph-comp>, and <graph-comp>2.attn.o:3:1000</graph-comp>) seem to fire inside or on angled brackets or on other markup and xml related closing and syntax elements like e.g. `","`, ‘`[@`...`]`’ and one appears to be active inside brackets in latex code.
- One subcomponent, labeled <graph-comp>2.attn.o:3:286</graph-comp> is active inside angled brackets, but also on what appear to be chat messages, with particularly high magnitude activations on the line breaks in these messages.
- Two subcomponents, labeled <graph-comp>2.attn.o:3:495</graph-comp> and <graph-comp>2.attn.o:3:121</graph-comp> appears to be more generally active active on contexts like latex, math, computer science, code and foreign language text.
- The final subcomponent is <graph-comp>2.attn.o:3:735</graph-comp>

<graph-page-break/>

**Layer 3 attention summary**
<graph-comp hidden>3.attn.q:3:334</graph-comp>
<graph-comp hidden>3.attn.v:3:120</graph-comp>
<graph-comp hidden>3.attn.k:1:145</graph-comp>
<graph-comp hidden>3.attn.v:0:677</graph-comp>
<graph-comp hidden>3.attn.v:1:677</graph-comp>
<graph-comp hidden>3.attn.v:1:76</graph-comp>
<graph-comp hidden>3.attn.v:1:95</graph-comp>
<graph-comp hidden>3.attn.o:3:283</graph-comp>
<graph-comp hidden>3.attn.o:3:398</graph-comp>
<graph-comp hidden>3.attn.o:3:806</graph-comp>

There are fewer subcomponents in the layer 3 attention of the graph than at the previous two layers. Judging by their labels, this layer attends to the `<` and `u` sequence position from the `v` sequence positions as part of generic previous token behavior.

This attention layer seems less crucial to the overall computation than layers 1 and 2. Ablating its attention output components, save for the one labeled <graph-comp>3.attn.o:3:806</graph-comp>, only lowers the probability on `>` from $0.547$ to $0.498$. Ablating this output "bias" subcomponent does essentially destroy performance — likely due to the central role of this component in setting typical activation sizes, since it has very high attributions to many downstream nodes, rather than any sophisticated computational role. Notably, the same is not true of the layer 2 attention output, which also has a subcomponent labeled <graph-comp no-highlight>2.attn.o:3:735</graph-comp>: ablating all output subcomponents in layer 2 of the graph save for that one still reduces the probability on `>` under adversarial sampling to less than $0.001$.

<graph-page-break/>

**Layer 3 attention matrices**

<graph-comp hidden>3.attn.v:0:677</graph-comp>

- There is only one query subcomponent on the `v` sequence position, labled <graph-comp>3.attn.q:3:334</graph-comp>
- There is one value subcomponent on the `v` sequence position, indicating that it is part of a self-attention mechanism in this layer: <graph-comp key>3.attn.v:3:120</graph-comp>. In slight contradiction to its autointerp label it also seems to sometimes activate slightly on tokens outside latex math mode, though only in the sort of text that may typically also feature latex, or on Latex-related tokens such as ` Eq`, ` Appendix`, ` proof`, and ` Newton`.
-  There are no key subcomponents on the `<` sequence position, and only one value subcomponent, <graph-comp key>3.attn.v:1:677</graph-comp>, which is also kept on the `u` sequence position. It is causally important on more than $25\%$ of tokens, firing mostly on delimiters, "syntactic glue words" like ` and`, ` the`, ` a`, ` is`, ` would`, ` of`, ` on`, ` to` and to a lesser extent text following right after delimiters and these connective words.
- There is one key subcomponent on the `u` sequence position, labled <graph-comp>3.attn.k:1:145</graph-comp>. This indicates that the relevant information is moved from `u` to `v` as part of generic previous token behavior.
- There are three value subcomponents on the `u` sequence position: The first, <graph-comp key>3.attn.v:1:677</graph-comp>, is also kept on the `<` sequence position. The two others are labeled <graph-comp>3.attn.v:1:76</graph-comp> and <graph-comp>3.attn.v:1:95</graph-comp>.
- There are two attention output components on the `v` sequence position, labeled <graph-comp>3.attn.o:3:283</graph-comp> and <graph-comp>3.attn.o:3:398</graph-comp>, and one subcomponent, labeled <graph-comp>3.attn.o:3:806</graph-comp>.

</graph-explanation>

Notably, in the attention layer 2 of this graph, information about the open bracket seems to be moved from the `<` position to the `v` position, partly due to information previously received from the `<` position in attention layer 1. This triggers a query that is specific to closing-delimiters (such as `>`), which searches for a preceding opening-delimiter (such as `<`) key.

Since the the $W_Q$ subcomponents used in attention layer 1 and 3 appear to be generically always-active rather than triggering in response to preceeding commas, and the queries in layer 2 do not appear to only trigger conditional on a comma at the previous token either, one might wonder whether the model would not also predict a closing `>` right after `u`. It turns out that it does — predicting `>` as its top logit after `u` as well, though with lower confidence ($0.119$ vs. $0.547$ after `v`). <ref>graph:bracket_u</ref> shows a structurally similar graph for this prediction, but lacking the components active on `u` and `,` in the attention layers. This suggests the longer context reinforces the math context and thus the likelihood of a closing bracket. Interestingly, the model does not predict a closing bracket after `,`, suggesting it recognises that the comma indicates the statement inside the bracket is not yet complete.

<label id="graph:bracket_u"/>

```graph
id: bracket-u-full
data: data/graphs/bracket-u-full.json
details: data/graphs/bracket-u-full-details.json
caption: Attribution graph for predicting `>` on the prompt `<` `u` `,` `v` `>` after `u`, pruned with adversarial sampling.<footnote>Coefficient $0.5$ for cross-entropy reconstruction with stochastic sampling, coefficient $0.5$ for cross entropy with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.1$, $p=0.3$, $4000$ optimization steps.</footnote> There are 162 subcomponents in the graph. The target model assigns probability $0.119$ to `>`.
```

Pruning with CI masking instead of adversarial masking recovers a much smaller graph of just 14 components (<ref>graph:bracket_ci</ref>). It predicts `>` correctly under CI masking but fails completely under adversarial masking, giving a very incomplete account of the computation. Nevertheless, it highlights some core pathways.

<label id="graph:bracket_ci"/>

```graph
id: bracket-minimal
data: data/graphs/bracket-minimal.json
details: data/graphs/bracket-minimal-details.json
caption: Attribution graph for predicting `>` on the prompt `<` `u` `,` `v` `>`, pruned with causal importance masking.
```


<graph-explanation name="bracket-minimal" >

<graph-comp hidden>output:3:31</graph-comp>

Attribution graph for predicting `>` after `v` on the prompt `<` `u` `,` `v` `>`, pruned with causal importance masking.<footnote>Coefficient $1.0$ for cross-entropy reconstruction with causal importance masking, importance minimality coefficient $0.1$, $p=0.3$, $2000$ optimization steps.</footnote> There are 14 subcomponents in the graph.

The two largest direct positive attributions to the output `>` come from:

1. A layer 3 MLP down projection subcomponent labeled <graph-comp>3.mlp.down:3:1414</graph-comp>. It receives attribution from one MLP up projection subcomponents labeled <graph-comp>3.mlp.up:3:2565</graph-comp> and one component labeled <graph-comp>3.mlp.up:3:1051</graph-comp>, with the latter firing on a more general set of closing delimiters.

2. <graph-comp>2.mlp.down:3:1560</graph-comp>, a layer 2 MLP down projection subcomponent. It connects strongly to the layer 3 MLP input component and subcomponent mentioned above in addition to the output, suggesting that the layer 2 and 3 MLP pathways are partially interlinked in series rather than parallel and independent. It receives attribution from an MLP up projection subcomponent labeled <graph-comp>2.mlp.up:3:2151</graph-comp>.

Ablating these two MLP down projection subcomponents out of the target model severely degrades the `>` prediction, lowering the probability from $0.547$ to $0.158$ and $0.243$ for individual ablations, and to $0.046$ for joint ablation. The model instead reassigns probability mass to other delimiters such as `)`, `_`, `,` or `)$`, suggesting they are important for singling out a right angled bracket in particular.

<graph-page-break/>
  
These MLP subcomponents receive information about the open angled bracket from layer 2 attention output components labeled <graph-comp>2.attn.o:3:855</graph-comp> and <graph-comp>2.attn.o:3:878</graph-comp>, which in turn receive from a component of the layer 2 attention value matrix  at the `<` sequence position labeled <graph-comp>2.attn.v:0:473</graph-comp> . This component receives information directly from the `<` embedding, as well as through three layer 0 MLP subcomponents, labeled <graph-comp>0.mlp.down:0:3069</graph-comp>, <graph-comp>0.mlp.up:0:2149</graph-comp>, and <graph-comp>0.mlp.up:0:2643</graph-comp>.

</graph-explanation>


While the 14-component graph highlights the core pathways, the full graph in <ref>graph:bracket</ref> makes clear that the actual computation is far more intricate. 


Given how few components our decomposition has in total (ca. 10,000 alive in the whole model) it is perhaps remarkable how many of them appear to be dedicated to moving around and processing information for predicting closing delimiters of various kinds. This may be partially due to delimiter closing being one of perhaps relatively few prediction tasks that is simple enough for a model of this size to perform well.



## Editing a language model's parameters by hand to modify its neural algorithm {toc: Editing a language model by hand}

<label id="sec:model-editing"/>


We used the decomposition to perform a simple edit to the model's learned algorithm: Manually modifying a single rank-1 component to make the model predict that all emoticons are surprised-face emoticons. Emoticons usually start with colons `:`, but colons are used in many non-emoticon contexts. The challenge here is to make models predict  the token `o`, as in a surprised-face emoticon `:` `o`, with high probability without substantially altering the model's behavior in other, non-emoticon contexts. Because `:` tokens can be used in many non-emoticon contexts, this rewrite can't be acheived with a token-level remapping; we have to rewrite the algorithm that the model applies to its hidden activations. 


We find that multiple subcomponents in the MLP out matrix of layer 2 specifically activate on the first characters in emoticons (e.g. `:`), with low activations elsewhere, including on these same tokens in other contexts:

- <comp key>2.mlp.down:2359</comp>: <comp>2.mlp.down:2359</comp>
- TODO(Lucius) subcomponent id of some of the other subcomponents
- TODO(Lucius) subcomponent id
- TODO(Lucius) subcomponent id

We picked one of these components, <comp key>2.mlp.down:2359</comp>, as our target for editing. Our edit leverages the idea that each subcomponent, being a rank-1 matrix $\vec{U^l_c} (\vec{V_c^{l}})^\top$ has one 'read' direction and one 'write' direction, which are its right and left singular vectors respectively. We changed the 'write' direction of the component so that, when it activates, it writes very strongly to the same direction as the `o` token in the model's unembedding matrix. We performed this edit by adding the unembedding vector for the `o` token, scaled by a prefactor of $\alpha$, to the $\vec{U_c}$ vector of the emoticon component.

<!-- (While this edit clearly affects the output in the intended way, there is another layer in between our edited layer and the output. Our edit might have some off target effects. We may be able to do better than this edit by choosing a direciton that maximally avoids affecting the computations of the intermediate layer while still projecting strongly onto the `o` token in the unembedding matrix. This may help to close the gap between the performance of our edit and the performance of the LoRA) -->

 
The resulting edited model assigns high probability to token `o`, whenever the emoticon component is causally important. 

<!-- TODO(Dan)(High priority): We can't make this claim without providing evidence. We should have a few generations maybe of the targeted model vs the edited model to show the edit basically works. If too much effort, maybe just reference the (currently unreferenced) figure fig:model-editing-heatmap -->
<!-- TODO(Dan) This section seems a bit jumbled because the ordering of VPD vs Lora sometimes seems swapped around in different locations. Wouuld be good to make consistent. WIll maybe require editing some panels (e.g. VPD should probably be on the left in fig:model-editing-heatmap)-->

To measure the amount of undesired off-target effects caused by the edit, we use two metrics, which characterize off-target effects in slightly different ways, one measuring effects on tokens that are potentially computationally 'nearby' to our edit, and the other measuring all changes: 

- $D_{\text{KL},\text{Surrounding}}$: The KL-divergence between the target model and the edited model on the $20$ tokens before and after a token on which <comp key>2.mlp.down:2359</comp> is causally important;
- $D_{\text{KL},\text{Global}}$: The KL-divergence between the target model and the edited model on all tokens on which <comp key>2.mlp.down:2359</comp> is not causally important, sampled from the whole dataset.

As baslines for comparison, we trained two conventional LoRA adapters for the MLP out projection matrix in layer 2. The LoRAs were trained to convergence on $n$ dataset examples ($n=10$ or $947$). The training dataset examples consisted of the token on which the subcomponent <comp key>2.mlp.down:2359</comp> is causally important and the 20 tokens before and after. They were trained both (a) to predict an `o` after the emoticon's initial token (e.g. `:`) and (b) to minimize the off target effects. Concretely, for (a), each LoRA was trained with a cross-entropy loss to predict the `o` label after the token on which the component is causally important. For (b), off-target effects were minimized using a KL divergence term (weighted by the off-target effect penalty coefficient, $\lambda$) between the output logits of the target model and the logits of the edited model on the rest of the tokens in the example<footnote>Here, the LoRA training loss is equivalent to both $D_{\text{KL},\text{Surrounding}}$ and $D_{\text{KL},\text{Global}}$ (due to how the datapoints are selected).</footnote>.


<figure class="wide">
<label id="fig:model-editing-pareto"/>
<img src="figures/editing_pareto.png"/>
<figcaption>Model editing for emoticon completions,  LoRA vs. manual subcomponent edit. Manual edits were performed by adding the unembedding vector for `o` to the $\vec{U}$ vector of the emoticon component with different prefactors $\alpha$. LoRAs were trained on $n=10$ and $n=947$ examples, each consisting of a token the emoticon subcomponent was causally important on, and the $20$ tokens immediately preceding and following it, with a KL-regularisation term weighted by $\lambda$. The y-axis shows the average probability the edited model assigns to `o` on tokens the emoticon component is active on. The x-axis in the left plot shows  $D_{\text{KL},\text{Surrounding}}$, the KL divergence between the edited model and the target model on the $20$ other tokens immediately preceding and following tokens the emoticon component is causally important on, across a holdout set of $50$ examples. The x-axis in the right plot shows $D_{\text{KL},\text{Global}}$, the average KL-divergence between the edited model and the target model on all other tokens across samples from the whole dataset.</figcaption>
</figure>



<label id="fig:model-editing-heatmap"/>
```heatmap
left_data: data/editing-kl-heatmap-lora.json
left_title: LoRA
right_data: data/editing-kl-heatmap.json
right_title: VPD
caption: Per-token KL divergence after editing to complete emoticons with `o`. On the left, KL-divergences for a model edited with a LoRA trained on $n=947$ examples, each consisting of a token the emoticon component was causally important on, and the $20$ tokens immediately preceding and following it. On the right, KL-divergences for a model edited by adding the unembedding vector for `o` to the $\vec{U}$ vector of the emoticon component with different prefactors $\alpha$.
```


In <ref>fig:model-editing-pareto</ref> we vary both $n$ (the number of training dataset examples) and $\lambda$ (the off-target effect penalty coefficient). We plot the trade-off between the probability of predicting an `o` versus off target effects. We compare the LoRAs with our manual edit with different scale factors $\alpha$ for the `o` unembedding vector added to the components’ $\vec{U}$ vector of <comp key>2.mlp.down:2359</comp>. 

LoRAs trained on just $n=10$ examples outperform the manual edit on $D_{\text{KL},\text{Surrounding}}$, the setting they were trained on, but not on $D_{\text{KL},\text{Global}}$. LoRAs trained with $n=947$ examples outperform the manual edit on both $D_{\text{KL},\text{Surrounding}}$ and $D_{\text{KL},\text{Global}}$.

While this is a promising result, we stress that this is a very preliminary investigation. The method we used to edit the component, adding the appropriate unembedding vector, was simply the first interpretable editing technique we tried. Other editing techniques might work better. For example, although this edit clearly affects the output in the intended way, there is another layer in between our edited layer and the output, which may lead to some of our edits' off target effects. We may be able to do better by choosing a directioon that maximally avoids affecting the computations of the intermediate layer while still projecting strongly onto the `o` token in the unembedding matrix. This may help to close the gap between the performance of our edit and the performance of the LoRA. 
<!-- Alternatively, we could try a hy

one could attempt a hybrid approach, training a LoRA with the right singular vector frozen to equal that of a component. This way, the edit might become more performant while remaining somewhat interpretable and data-efficient.  -->
On the other hand, the example is cherry picked. We deliberately chose this task because the model seemed to have a small number of components related exclusively to emoticon prediction. 
We nevertheless conclude that VPD shows some promise for model editing in cases where correctly labeled data for training a LoRA is difficult to obtain, or where it is desirable for the edit to be somewhat interpretable. We think that there is very likely ways to leverage parameter decomposition to do much better editing than we have here in this proof of concept.

<!-- #### All emoticons end in o -->



## Discussion

<label id="sec:discussion"/>

<!-- Note from Lee: I've moved the majority of the discussion and its notes to the discussion-drafting.md replit file. That file is approximately frozen. It contains all the text that used to be here. I'm live-drafting a version in post.md below. -->

At this point, it's worth reflecting on what our parameter decomposition approach has actually bought us with regard to the highest-level goals of mechanistic interpretability. 

In mechanistic interpretability, we want to reverse engineer the computational machinery of neural networks. In particular, we want to know how that machinery takes inputs, computes hidden representations, performs computations on those hidden representations, and computes their output behavior. Concretely, this means that the object we want to understand is a neural network's computational graph and how it interacts with data. To the extent that it is possible, we'd like to understand small, bite-sized parts of the network's computation, and have our explanations aggregate together so that, eventually, we could understand the entire network. And to make this as manageable as possible, we'd to achieve this using as short a description as possible. 

We'll analyze how VPD intersects with these goals in some detail in the following sections. 

### Parameter decomposition makes fewer assumptions about neural network's representations than other methods

<label id="sec:discussion-computation"/>

In a way, parameter decomposition methods are a less 'opinionated' approach to mechanistic interpretability compared with other popular decomposition methods such as SAEs or CLTs. They are less opinionated than other methods about the 'form' of the comptuation that we expect to find. Most sparse dictionary learning methods, for example, make the implicit assumption that neural computation might parsimoniosly be described using graphs of thresholded linear functions. But it remains unclear whether explanations of this form will be parsimonsious, and there are some indications that they are not. 

If not, then it may be prudent to take a less opinionated approach to decomposition, letting the causal structure of the model itself tell us what form it computations might take. This overcomes the issue of having to choose an arbitrary nonlinear function to represent a given network's computations, such as the thresholded linear function used in PLTs and CLTs. Indeed, the nonlinear threshold used in some transcoders may not even be used in the original model, making it hard to justify on principled grounds. Parameter decomposition avoids this issue by using the target network's computational graph, rather than training another model that performs similarly to the target model on the training data, but may not be mechanistically faithful, implying somewhat different behavior off distribution.

### Explanations of attribution graphs are not explanations of computational graphs {toc: Attribution graphs are not computational graphs}

A full explanation of a network's behavior should essentially amount to an end-to-end algorithm that is essentially equivalent to the algorithm implemented by the target network. In other words, it should be possible to represent the explanation as a *computational graph* that is mechanistically faithful to the original computational graph of the network, typically expressed in terms of its neurons, weights matrices, nonlinearities, etc. 

In this paper, we used VPD to produce *attribution graphs* rather than computational graphs. It is not possible to compute the model output on a datapoint using only the attribution graph without access to the original model itself.  An attribution graph can track how strongly any given upstream node in a computational graph influenced any given downstream node, which is useful for understanding the flow of information in the graph, but it does not represent the functional relationship between upstream and downstream nodes. This means the explanations of the model's behavior we provide are incomplete. We have not yet explained the network's computational graph. But we have nonetheless identified a set of objects that are faithful, minimal, and simple, which is an important step toward a full explanation of network behavior that is as parsimonious as possible.

Additionally, attribution methods such as the gradient attributions we used in this paper also have some well-known issues that can lead them to misjudge the magnitude of the influence one node in a graph has on another <cite>neel2022attribution</cite>, <cite>kramár2024atpefficientscalablemethod</cite>. For example, if an attention head in a model has a saturated softmax, gradient atrributions through it will tend to systematically underestimate the effect of ablating the upstream node on the downstream node.

Despite these limitations, we think attribution graphs are still useful as a basic picture of how information flows between VPD components on a forward pass, and have been used to similar effect for other decomposition methods, such as CLTs <cite>ameisen2025circuit</cite>. In future work we aim to deepen our study of full computational graphs by studying in detail the interactions of VPD components at nonlinearities, such as MLP neuron activation functions. For some preliminary investigations into characterising nonlinear interactions between subcomponents at MLP neurons, see <ref>app:interactions-gis-vs-coact</ref>.



### Robustness to (adversarial) ablations permits aggregation of explanations

As discussed in <ref>sec:vpd_recon_motivation</ref>, one of the central promises of ablation-based parameter decomposition is that explanations of a model's behavior on individual datapoints, given in terms of causally important parameter components and their interactions, can be aggregated into more global explanations of its behavior across the full distribution (<ref>sec:mech-faith-aggregate</ref>). In principle, one could start from individual datapoint explanations and incrementally combine them — first into explanations of the model's behavior on narrow sub-distributions (such as bracket closing or pronoun prediction), then into broader and broader accounts, eventually approaching a complete reverse engineering of the model. We have not yet attempted this aggregation in practice, and whether our current decomposition is ultimately sufficiently adversarially robust for this purpose remains unclear. Our primary uncertainty is that it is unclear how much adversarial robustness is necessary for 'local' explanations to aggregate into 'global' ones.

<!-- We stated in <ref>sec:opt-mech-faithfulness</ref> that we would ideally like our decomposition to be robust to ablating any combination of parameter components that are marked as causally We can first find explanations for the model's behavior on individual data points, each involving small subsets of components, then aggregate them into larger subsets of components that explain the model's behavior over sub-distributions (e.g. a single subset of components for all bracket closing tasks and a different subset of components for all pronoun prediction tasks), then incrementally aggregate those with each other to slowly understand the whole target model's behavior under the full data distribution.  -->

<!-- For example, we'd like to be able to study different sets of components for different bracket closing and pronoun prediction prompts, then --> 

If we do not have enough adversarial robustness, then we lose the ability to aggregate explanations of parts of the model into a coherent whole. However, if we are too strict in our demands for robustness to adversarial ablations, it is sometimes possible to exclude decompositions we would intuitively regard as valid, because the adversary can systematically exploit random interference noise in 'unused' circuitry to change the network output. In <ref>sec:vpd_methods-adv</ref>, we point out a theoretical toy case in which too strict a demand for adversarial robustness causes this problem. This seems to put us in a somewhat difficult spot. How much robustness do we need to demand for our explanation to be mechanistically faithful? How much robustness is actually too much, and would exclude short descriptions of network behavior we would like to regard as valid? 

We do not currently have a fully satisfying answer to this question, but we suggest that a reasonable approach may be to ground the answer in practical considerations: What combinations of (sub)component ablations might we realistically want to perform when using VPD to understand or edit a given model? And over which subsets of the data would we want to investigate the behavior of the resulting ablated models? So long as the decomposition is robust enough that it is unlikely for any of the ablations we end up performing in practice to be in the non-robust set for any model input we care about, the lack of complete robustness may not be relevant to us. Even if we do end up encountering a component ablation the decomposition is not robust to, the problems caused by this may be limited if they only apply to a few data points and the edited model is still behaving as we would expect for the vast majority of inputs<footnote>Ultimately, we think this practical mindset also sheds some light on how we can think about the theoretical toy case we mentioned above, where the adversarial sampler exploits unstructured noise in 'inactive' circuits to change the output: Those 'inactive' circuits really are *somewhat* involved in computing the model's outputs. It's just that their involvement is quite limited, and only becomes relevant in exponentially rare edge cases that involve ablating very particular sets of components on very particular data points. So, a shorter description of the model's behavior in terms of the much small number of 'active' components really isn't completely mechanistically faithful, but it is *mostly* mechanistically faithful. Dropping the inactive circuits from the description retains its predictive power for almost all cases we could possibly care about while drastically decreasing its length, and that trade-off is usually worth it to us.</footnote>. 





### Limitations

<label id="sec:limitations"/>


**Our decomposition is not as adversarially robust as we would like.** As shown in <ref>tab:vpd-pgd-ce</ref>, while the decomposition is at least somewhat robust to $\approx 20$ steps of adversarial optimization (KL divergence $0.83$), robustness degrades rapidly with more optimization steps, reaching a KL divergence of $40.2$ at $320$ steps. This means that there exist sets of subcomponent ablations involving only causally unimportant components that drastically alter the model's output. As discussed in <ref>sec:discussion-robustness</ref>, we do not necessarily expect or even desire complete robustness to arbitrarily many steps of adversarial optimization. However, a previous decomposition we performed on an even smaller language model suggests that substantially higher levels of adversarial robustness ought to be achievable.

**Our simplicity measures are imperfect and ad-hoc.** The rank constraint and frequency minimality loss ($\mathcal{L}_{\text{frequency-minimality}}$) we use are just one possible set of proxies for encouraging parameter components to be computationally simple objects, and we have no reason to believe that they are the optimal choice. 
<!--While we showed that it addresses the failure mode in which multiple rank-one mechanisms can sum to another rank-one matrix (<ref>fig:simplicity</ref>), the loss is motivated by information-theoretic intuitions rather than a rigorous definition of what "simplicity" should mean for a parameter component. There may be other failure modes we have not yet identified. More broadly, we lack a principled, general-purpose measure of the computational simplicity of a parameter component, and developing one remains an open problem.-->

**Our clustering method is blind to multi-sequence position circuits** VPD decomposes weight matrices into rank-one subcomponents, which must then be clustered into full parameter components that span multiple weight matrices (<ref>app:clustering</ref>). Our clustering algorithm is based on minimising description length, but it currently only uses correlations between causal importances on the same sequence position. This ignores possible compression based on cross-sequence position correlations. For example, $Q$ and $K$ components in an induction head might never operate on a computation at the same sequence position. 

**Our clustering method has not been carefully tuned.** Our MDL-based clustering algorithm has a key hyperparameter $\alpha$ that controls the trade-off between the number of clusters and their complexity. We did not sweep this hyperparameter particularly carefully. This was not a priority because individual subcomponents already proved to be fairly interpretable on their own, but it means the parameter components we report may not reflect the best possible grouping.

<!-- Perhaps relatedly, we observe that the weight vectors of subcomponents within the same cluster tend to have rather low cosine similarity with each other across different weight matrices. (N.b. Lee: I'm not actually sure how true this is. They have better similarity than two seeds of unclustered Jose!  ) -->

  **We present attribution graphs, not computational graphs.** As discussed in <ref>sec:discussion-graphs</ref>, the attribution graphs we use to study circuits of parameter components (<ref>sec:circuits</ref>) are necessarily incomplete descriptions of the model's computation. They track how strongly upstream components influence downstream components on a given forward pass, but they do not represent the functional relationships between them. It is not possible to compute the model's output from an attribution graph alone without access to the original model. Furthermore, gradient-based attributions have well-known failure modes — for  example, saturated softmax functions in attention layers can cause gradient attributions to systematically underestimate the true causal effect of upstream components. Moving to full computational graphs that represent the nonlinear interactions between parameter components at MLP neurons and other nonlinearities is an important direction for future work (<ref>sec:future-work</ref>). Our preliminary analysis suggests that parameter components may tend toward simpler nonlinear interactions than might be feared (<ref>app:interactions-gis-vs-coact</ref>), which is perhaps somewhat encouraging for the feasibility of this direction, but it is still far from definitive evidence.


### Future work

<label id="sec:future-work"/>

<!-- 
- Better sampling strategies (improving adversarial sampling, other parameterisartions of masking space). E.g. our PPGD sampler is still quite primitive. 
- Improve causal importance function performance. Right now this is just a vanilla transformer which is fed the original model's hidden activations concatenated in a big vector. Probably one can do better than this.
- We want these improvement so that we can apply VPD to bigger llms. We also want to apply VPD to other models such as biological foundation models.
- Computational graphs instead of attribution graphs. E.g. learn simple functions of mlp output component activations in terms of mlp input component activations. (Later, maybe something more mechnaistically faithful like single neuron transcoders. Note to Claude: don't try to understand that last part, it's an idea we haven't talked about here yet.)
- More autointerp for fully reverse engineering computational graphs of the model on particular data points, instead of just studying some parts of them the way we did with the attribution graphs here.
- Aggregating explanations for individual datapoints to reverse engineer a small language model completely.
- VPD applied to subsets of the data instead of the whole dataset. This won't capture all the components, just those relevant for that data subset, but might be much cheaper and more convenient
- More advanced model editing than what we did in the model editing section here. E.g. generating multi-step conditional circuits by modifying component $V$ vectors to take in activation from other components they originally didn't connect to strongly. Hybrid approaches using LoRA-like training restricted to particular right or left singular vectors of components.
- Using VPD for interpretable gradients. Components are just directions in parameter space, so one can transform network weight gradients into the component basis of the subspace they span. This may let us interpret every gradient in terms of upweighting, downweighting or modifying existing components and creating new components. And since components are interpretable, this gives us some idea of what each update is teaching the model. That way, weight updates may perhaps be monitored during training.
- Our attention calculations are interesting, but we described a behavior that is 'always on'. We would like to know more about behaviors that are distributed across heads that are more conditional.

Lightly edited Claude draft:-->

**Improved adversarial sampling and mask parameterizations.** Our current adversarial sampler uses a relatively primitive form of projected gradient descent (PGD) to find worst-case ablation masks. We think it should be possible to improve the performance of this sampler. For example, we might be able to identify particularly important subspaces of masking space for the sampler to focus on, such as the subspaces spanned by the causally important components on other data points in the same batch.

**Better causal importance functions.** The causal importance function $\Gamma$ is currently implemented as a vanilla transformer that takes as input the target model's hidden activations concatenated across layers into a single vector. This is a relatively simple architecture for a task that requires predicting the ablatability of every subcomponent at every sequence position, and we suspect that more sophisticated architectures might produce more accurate causal importance predictions. 
<!--Improving the causal importance function would directly improve the quality of the decomposition by producing better ablation masks during training.-->

**Continuous cut-off scales instead of binary causal importances.** Our causal importance functions currently classify subcomponents in a largely binary manner: Either they are causally important for computing the network's output, or they are not. However, in reality, subcomponents lie on a more continuous scale of affecting the output to a larger or smaller degree. The more we care about low description length relative to output reconstruction, the more subcomponents we will want to drop from our description of the forward pass, starting with those that affect the final output the least. To account for this, we might train a function that predicts *cut-off scales* on the pareto frontier between output reconstruction and description length instead of fixed causal importances. This way, a single decomposition could provide a variable resolution scale for describing the forward pass, ranging from short and simplified descriptions of the network's computation involving just the most important components, to longer but more accurate descriptions involving more components, all the way up to descriptions  which recover the target model's performance completely.<footnote>The causal importance functions already enable this to an extent through the fractional causal importance values, see <ref>fig:pareto-mse</ref>, but they are not really trained with this application in mind.</footnote>

**Scaling to larger models and non-language models.** The improvements to adversarial sampling and causal importance functions described above are partly motivated by the goal of applying VPD to larger language models than the 67M-parameter model we decomposed here. Beyond scale, we are also interested in applying VPD to other domains, such as biological foundation models.
<!--, where mechanistic understanding of learned representations could be particularly valuable.he decomposition by producing better ablation masks during training.-->

**From attribution graphs to computational graphs.** As discussed in
<ref>sec:limitations</ref>, the attribution graphs we use to study circuits provide an incomplete picture of the model's computation: they summarize local interaction strengths between components but do not represent the functional relationships. A key direction for future work is to move toward full computational graphs that explicitly represent how components interact at nonlinearities. One concrete approach would be to learn simple functions that predict MLP output subcomponent activations in terms of MLP input subcomponent activations, capturing the nonlinear transformation performed by intermediate neurons. 
<!-- Oli: I think the previous sentence is begging for a comparison to transcoders -->

**Automated reverse engineering of model computations.** In our case studies (<ref>sec:case-studies-pronoun</ref>, <ref>sec:case-studies-bracket</ref>), we manually traced information flow through small parts of the attribution graphs for a few specific prompts and behaviors. Building a full picture of how a model computes its outputs will require scaling up this kind of analysis considerably, to more prompts and on more paths through their graphs. We aim to do this using automated interpretability methods.

**More advanced model editing.** Our model editing experiment (<ref>sec:model-editing</ref>) demonstrated a proof-of-concept in which we modified a single rank-one component's left singular vector to change the model's emoticon predictions. More ambitious editing could go further in several directions: for example, one could attempt to create deeper behaviors with multiple conditions in series by modifying subcomponents' right singular vectors $\vec{V_c}$ to connect more or less to the left singular vectors $\vec{U_c}$ of other components. Hybrid approaches that combine the interpretability of parameter components with the optimization power of LoRA — for instance, training a low-rank adaptation with left or right singular vectors restricted to those of specific components — could also yield edits that are both more performant and more interpretable than either approach alone.

**Data-subset decompositions.** Rather than decomposing the model with respect to the full training distribution, one could apply VPD to a specific data subset, recovering only the components relevant to that subset. This would not surface all the model's components, but it might be substantially cheaper and more practical for more narrowly targeted investigations or editing.

**Interpretable parameter gradients.** Parameter components are directions in parameter space, so it is possible to project any parameter gradient into the basis defined by the decomposition's subcomponents. This could allow us to express each gradient update to a model as a combination of upweighting, downweighting, or modifying existing components, as well as creating new ones outside the span of the existing component subspace. Since individual parameter components are interpretable, this may give us some idea of what each training step is teaching the model.
<!--This is importantly different from what's possible with activation-based interpretability methods, while activation gradients can be projected onto directions in activation space, without an interpretable basis for parameter space, gradients cannot be understood in terms of how they alter mappings between representations, only the representations themselves.-->


<!-- todo(Lucius)(Low priority): DIsCUSS see if we want to add any ideas from January's project list: https://docs.google.com/document/d/1JuUzQyjASxEo_m_F23rJVkIrozhhPW-l8FimoScURfg/edit?tab=t.0  -->

<!-- todo(Lucius)(Low priority): DISCUSS Add mention that we haven't really looked at what's going on with RMS norm. Lucius: Not sure what you mean. Lee: It's a nonlinearity that is used in our model but we haven't studied how VPD interacts with it, unlike the elementwise nonlinearities in the MLP (as in our interaction exp) and attn. -->


### Related work

<label id="sec:related-work"/>

#### Ablation-based parameter decomposition

VPD is built primarily on prior parameter decomposition methods, namely attribution-based parameter decomposition (APD) <cite>braun2025interpretabilityparameterspaceminimizing</cite> and stochastic parameter decomposition (SPD) <cite>bushnaq2025spd</cite>. These papers introduced most of the core ideas used by our method, including (a) the idea that networks could be decomposed into sparsely used functional units consisting of vectors in parameter space that sum to the parameters of the target model, and (b) causal importances can be identified using a causal importance network and ablations. SPD lacked adversarial sampling scheme that would make the causal importances robust to adversarial ablations, as well as the additional loss to encourage computational simplicity, here implemented as the frequency-minimality loss. Those works also focused primarily on toy models, rather than language models trained on natural data. Other work <cite>christensen2025decomposition</cite> did apply SPD to parts of a larger model, but did not decompose a whole language model, and lacked the crucial extra losses as Bushnaq et al <cite>bushnaq2025spd</cite>.


#### Identifying computational subgraphs in architectural unit basis

Much work in interpretability views neural networks as computational graphs and circuits as computational subgraphs that have a particular function <cite>wang2022interpretability, conmy2024towards</cite>. The identification of subgraphs has been approached through a range of methods, including using learned masks, ablations, or the use of attributions to identify ablatable network components. 

Some of the work that identifies subgraphs learns explicit differentiable masks <cite>csordás2021neuralnetsmodularinspecting, decao2021sparseinterventionslanguagemodels</cite> is loosly analagous to our causal importance functions. But these methods use the learned masks as the actual ablations, rather than to parameterize an ablation procedure. It is very unlikely, therefore, that the masks are robust to adversarial ablation (where, e.g. the masked parameters are only partly ablated, which should be equivalent to full ablation if those parameters were actually causally important) and hence unlikely that the 'subnetworks' found by those works are mechanistically faithful. Those works also learned masks for sets of datapoints, rather than single datapoints, as in our work. Additionally, the masks learned by those works were aligned with the parameter unit basis, unlike in our work where the parts of the parameters that are ablated are not necessarily aligned with the parameter unit basis. Later work <cite>conmy2024towards</cite> adapted the mask-learning procedure of <cite>decao2021sparseinterventionslanguagemodels</cite> to identify subgraphs where each node could be tested for its importance on a task, which is assessed by ablations, namely activation patching. Activation patching involves replacing a nodes activation with a choice of baseline, such as the zero, mean, random, or other baseline. Our work operates on parameters, and therefore avoids the need to choose a baseline in activation space. 

#### Identifying computational subgraphs using learned decompositions

Much of the above work operates on architectural components of networks, such as the neuron unit-basis, parameter unit-basis, whole MLP layers, or whole attention heads <cite>csordás2021neuralnetsmodularinspecting, decao2021sparseinterventionslanguagemodels, conmy2024towards, wang2022interpretability</cite>. But neural computations may not be aligned with those bases, and therefore the subgraphs they identify may involve components that are polysemantic (cite polysem references) and thus not yield accounts of neural computation that are maximally parsimonious. Like our work, existing work aims to address this issue by learning decompositions of neural networks from which to make more easily interpretable subgraphs (though see cite transluce paper, which argues that the neuron basis was not as unparsimonious as previously thought). 

Most similar to ours is the line of work that involves training CLTs and building attribution graphs for them, thus enabling accounts of computation that are not necessarily aligned with individual neurons or layers <cite>ameisen2025circuit, lindsey2025biology, kamath2025tracing</cite>. CLTs build on per-layer transcoders <cite>dunefsky2024transcodersinterpretablellmfeature, ge2024localglobal</cite>. In contrast to our work, CLTs and transcoders decompose activations, which are the results of computations, rather than parameters, which learn to implement the computations (through interactions with the nonlinearities). Additionally, while Kamath et al. <cite>kamath2025tracing</cite> built on CLTs to extend their attribution graphs to attention layers, their approach did not identify ways to decompose attention layers into functional units that may be distributed across heads. In our work, our parameter subcomponents learn specialized functional roles and also span multiple heads by default.

In addition to these topics, our work builds on broader foundations, including sparse dictionary learning, causal mediation analysis, interpretability of neural network parameters, automated circuit discovery, and other topics. We refer readers to our previous papers for deeper discussion of prior work on related topics <cite>braun2025interpretabilityparameterspaceminimizing, bushnaq2025spd</cite>.

### Conclusion

<!-- Rough points:

- Lots of progress since SPD. 
- Great that it's no longer on toy models only.
- We're not confident that the method won't need further changes, but seems good and pretty scalable for now.
- Exciting that it basically works
- Understanding parameters feels within reach
- Understanding parameters is a harder task than learning a replacement model - you don't get to be so opinionated about the type of computation you're expecting to find. It's not all lookup tables. Significant analysis might need to go into understanding some computations, like computation on manifolds. But we've now got a set of objects that serve as a good starting point for that. And while they haven't solved the entire problem of mech interp, they have solved is as much as, and seem basically as useful as, other methods anyway!
- Dream: Being able to use our deeper understanding of parameter components in order to be able to write whole networks, whole minds, that have more of the qualities we want (esp. safety), and less of the qualities that we don't. This feels like a good step toward that vision. -->

On the surface, neural networks' computations seem like monolithic, irreducible transformations; their parameters seem like large inscrutable matrices of floating point numbers. Parameter decomposition offers a lens with which these matrices can be decomposed and their computational roles scrutinized. We are very excited that now, with VPD, it is possible to decompose non-toy models (such as language models) that can solve tasks using neural algorithms that we do not know how to design ourselves. This represents an important step beyond the capabilities of previous parameter decomposition methods <cite>bushnaq2025spd, braun2025interpretabilityparameterspaceminimizing</cite>. However, it remains likely that the method requires further improvement for future work, such as scaling the method to larger models, or to address unforeseen pathologies with the current method. Even if key parts of the method require rethinking, we believe future iterations of the method will continue to resemble VPD in spirit. 

<!-- Moved to discussion: (TODO(Lee) - find a new way to wrap up the conclusion now that this content is moved) In some ways, parameter decomposition methods are a less 'opinionated' approach to mechanistic interpretability compared with other popular decomposition methods such as SAEs or CLTs. They are less opinionated than other methods about the 'form' of the comptuation that we expect to find. For example, most sparse dictionary learning methods make the implicit assumption that neural computation might parsimoniosly be described using graphs of thresholded linear functions. But it remains unclear whether explanations of this form will be parsimonsious, and there are some indications that they are not. If not, then it may be prudent to take a less opinionated approach to decomposition, letting the causal structure of the model itself tell us what form it computations might take. We take this approach with parameter decomposition. -->
 And while the approach does not yet solve all major problems in mechanistic interpretability, we have shown that it can be used any of the major interpretability tasks (such as constructing interpretable attribution graphs for circuit analysis) that have so far been achieved with other methods, such as CLTs. 

We think parameter decomposition may open up new affordances, not just for mechanistic interpretability, but for deep learning in general. We need to understand neural algorithms in terms of their parameters before we can unlock the ability to design (either by hand; in an automated way; or by informing our choice of training data) whole neural networks, whole minds, that have more of the qualities that we want, and less of the qualities we do not. By enabling decomposition of network's parameters into mechanistically faithful, minimal, simple parts, we think VPD represents an exciting step toward that vision. 

 


## Contributions statement

<label id="sec:contributions-statement"/>

#### Research iteration

Our method underwent significant iteration throughout development, changing many times in response to experimental results. LB, OCG, LS, and DB were primarily responsible for driving forward various iteration cycles, with NH responsible for some cycles. DB and LB tuned hyperparameters for various methods throughout the length of the project. LB did early method and hyperparameter iteration to get adversarial losses working on toy models and an earlier model trained on SimpleStories.

#### Conceptualisation

LB conceptualised the adversarial reconstruction loss and its implementation via projected gradient descent (PGD) on sources, with some input from LL. OCG came up with using persistence in the adversarial training loss and did hyperparameter optimisation for it. DB conceptualised the part of the current adversarial loss which does several steps of warmup of the persistent sources for each outer loss step.
LS identified the pathological bisemanticity of component activations that helped to motivate the addition of a 'computational simplicity' penalty. LB, based on discussions with LL and external collaborators as well as empirical iteration, conceptualized the frequency-minimality loss and did most of the testing and tuning for it.
LB conceptualized the new lower-leaky sigmoid after discussion with LS. LL conceptualised the sign exception on the straight-through estimator after LB noticed a problem with the previous version.
LB conceptualized delta components and did the early testing for them. NH came up with the idea for subset routing and ran the first experiments with it. LS conceptualized the parameter faithfulness warmup and did some experimental investigation into its usefulness. NH also contributed p-annealing and other method optimizations and evaluations that were useful for assessing the value of modifications to the method.
OCG designed the current causal importance function architecture, as well as the shared_mlp, global_shared_mlp, and vector gate MLP architectures used in earlier versions. LS did an initial implementation of the global causal importance function. LB conceptualised post-hoc causal importance optimisation and post-hoc adversarial optimisation restricted to base graph nodes, and did most of the hyperparameter tuning for post-hoc causal importances.
NH contributed p-annealing, subset reconstruction losses, and other methods optimizations.
LB conceptualised using component activations on top of causal importances for interpretability.

#### Clustering

LB conceptualised the first form of the clustering algorithm, including the MDL framing, initial MDL loss function, hierarchical merging, stopping based on MDL minimum, and picking alpha based on coactivation threshold. MI developed the algorithm further, with inputs from NH, LB, and LS. NH helped MI on clustering, primarily conceptually. LB did some of the empirical iteration to pick a clustering for the paper. OCG and DB optimized the clustering implementation for efficiency.

#### Attributions and analysis

LB did much of the conceptualisation work for the attributions used in the paper (including gradient stopping), with input from OCG, DB, and LS. LS conceptualized the dataset attributions.
LS and LB jointly conceptualised the nonlinear interaction metric. LS ran initial investigations into nonlinear interactions on an older language model, and LB ran the nonlinear interaction experiments used in the paper.
LS was responsible for the analysis of attention behaviors and the geometric consistency seed analysis.
LB did the first biostory on the simple stories model and two of the biostories in this paper.

#### Model editing

OCG did early explorations of model editing. LB contributed early conceptualisation for model editing. OCG and LB together did the final version of the model editing experiment in the paper.

#### Comparisons and evaluations

OCG was primarily responsible for autointerp pipeline and intruder detection comparisons.
BB trained the per-layer and cross-layer transcoders used for comparisons to VPD, did the evaluation and analysis of the reconstruction performance comparing VPD to transcoders, and did the feature splitting analysis.

#### Target Model pretraining

DB was responsible for model pretraining. LS helped train target models on the Pile dataset.

#### Engineering and infrastructure

OCG and DB equally managed the codebase and the implementations of the various methods.

#### Visualization and interactive figures

OCG was primarily responsible for the internal visualization app and for the interactive figures in the paper. DB helped with the internal visualisation app and the attribution graph visualisation. LS and LB contributed some features to the visualization app. LS designed and made various didactic figures used in the paper.

#### Writing

LS planned the paper and wrote initial drafts of some sections. LB wrote initial drafts for the two biostories, methods sections on frequency minimality loss, mechanistic faithfulness, and adversarial loss, the nonlinear interactions section, model editing section, parts of the discussion section, training recipe, and most of the mathematical sections in the appendix. MI wrote an initial draft of the paper section on clustering. BB drafted the section comparing VPD to transcoders and drafted the feature splitting section. OCG was primarily responsible for web development and for the interactive figures, with contributions from others. DB helped with editing.

#### Project management and mentorship

LS was responsible for overall management of the project and planning the paper. LS was the main point of contact for MI, NH, and BB and gave input on their work throughout the collaboration. LB also gave input on their work.












## VPD method details

<label id="sec:vpd_methods"/>

Here, we expand on some aspects of adVersarial Parameter Decomposition (VPD) in more detail. See <ref>sec:method</ref> for an introduction to VPD.


### $\Delta$-L2 penalty

<label id="sec:vpd_delta_l2"/>

The $\Delta$-components are different from normal subcomponents we train. Their rank can be greater than $1$, meaning they can be more complicated objects than regular subcomponents. We thus have a particular interest in ensuring that they are not used to compute the model's outputs. Theoretically, since we define the causal importances of Delta-components to always be zero, the stochastic and adversarial losses should ensure that this is the case. But in practice our reconstruction losses are not perfect, so we additionally encourage the $\Delta$-components to be exactly zero with an auxiliary mse loss:

$$\mathcal{L}_{\text{Delta-L2}}=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}\left(\Delta^l_{i,j}\right)^2=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}{\left( W^{l}_{i,j}- \sum^C_{c=1} U^l_{i,c} V^l_{j,c}\right)}^2.$$
<!-- LaTeX original:
\mathcal{L}_{\text{Delta-L2}}=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}\left(\Delta^l_{i,j}\right)^2=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}{\left( W^{l}_{i,j}- \sum^C_{c=1} U^l_{i,c} V^l_{j,c}\right)}^2.
-->
Here, $N$ in the total number of model weights.

### Causal Importance Function Architecture {toc: Causal Importance Function}

<label id="sec:vpd_ci_function"/>

The causal importance function $\Gamma$ maps the target model's hidden activations to
per-subcomponent causal importances. It is a single, shared network that jointly computes causal
importances for all subcomponents across all weight matrices in the target model.

**Inputs.**

Let $L$ denote the number of weight matrices being decomposed, and let $h^l_{b,t} \in
\mathbb{R}^{d_l}$ denote the input hidden activation to weight matrix $l$ of the target model at batch element $b$ and
sequence position $t$. Each activation vector is independently RMS-normalized, and the
normalized vectors are concatenated to form the input:

$$h_{b,t} = \left[
 \operatorname{RMSNorm}(h^1_{b,t}) \;|\; \cdots \;|\;
 \operatorname{RMSNorm}(h^L_{b,t})
 \right] \in \mathbb{R}^{D},
 \quad D = \sum_{l=1}^{L} d_l.$$
<!-- LaTeX original:
a(x, t) = \left[
 \operatorname{RMSNorm}(h^1_{b,t}) \mathbin{\|} \cdots \mathbin{\|}
 \operatorname{RMSNorm}(h^L_{b,t})
 \right] \in \mathbb{R}^{D},
 \quad D = \sum_{l=1}^{L} d_l.
-->

**Input projection.**

The concatenated activation vector is linearly projected to the transformer's $d_{\mathrm{model}}$ dimension:

$$z^{(0)}_{b,t} = W_{\mathrm{in}} z_{b,t} + b_{\mathrm{in}},
 \quad W_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}} \times D}, \;
 b_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}}}.$$
<!-- LaTeX original:
h^{(0)}(x, t) = W_{\mathrm{in}} a(x, t) + b_{\mathrm{in}},
 \quad W_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}} \times D}, \;
 b_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}}}.
-->

**Transformer layers.**

The projected activations are processed by $N$ pre-norm transformer layers. Each layer $n \in
\{1, \ldots, N\}$ applies bidirectional multi-head self-attention followed by a feedforward
network, each with a residual connection:

$$
\begin{aligned}
\hat{z}^{(n)}_{b,t} = z^{(n-1)}_{b,t} +
\operatorname{Attn}\!\left(
\operatorname{RMSNorm}\!\left(z^{(n-1)}_b\right)
\right)\!(t), \\
z^{(n)}(x, t) = \hat{z}^{(n)}_{b,t} +
\operatorname{FFN}\!\left(
\operatorname{RMSNorm}\!\left(\hat{z}^{(n)}_{b,t}\right)
\right),
\end{aligned}
$$
<!-- LaTeX original:
\hat{h}^{(n)}(x, t) &= h^{(n-1)}(x, t) +
 \operatorname{Attn}\!\left(
 \operatorname{RMSNorm}\!\left(h^{(n-1)}(x, \cdot)\right)
 \right)\!(t), \\
 h^{(n)}(x, t) &= \hat{h}^{(n)}(x, t) +
 \operatorname{FFN}\!\left(
 \operatorname{RMSNorm}\!\left(\hat{h}^{(n)}(x, t)\right)
 \right),
-->

where $\operatorname{Attn}$ denotes multi-head scaled dot-product attention with Rotary
Position Embeddings (RoPE) <cite>su2024roformer</cite>, applied bidirectionally (i.e., without a
causal mask) across all $T$ sequence positions; and $\operatorname{FFN}$ is a two-layer
feedforward network with GELU activation:

$$\operatorname{FFN}(z) = W_2 \operatorname{GELU}(W_1 z + b_1) + b_2,
 \quad W_1 \in \mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}}, \;
 W_2 \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}.$$
<!-- LaTeX original:
\operatorname{FFN}(z) = W_2 \operatorname{GELU}(W_1 z + b_1) + b_2,
 \quad W_1 \in \mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}}, \;
 W_2 \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{ff}}}.
-->

**Output head.**

After the final transformer layer, a linear output head projects back to the total number of
subcomponents:

$$z^{(N+1)}_{b,t} = W_{\mathrm{out}} z^{(N)}_{b,t} + b_{\mathrm{out}},
 \quad W_{\mathrm{out}} \in \mathbb{R}^{C_{\mathrm{total}} \times d_{\mathrm{model}}}, \;
 C_{\mathrm{total}} = \sum_{l=1}^{L} C_l.$$
<!-- LaTeX original:
\Gamma(a(x,t)) = W_{\mathrm{out}} h^{(N)}(x,t) + b_{\mathrm{out}},
 \quad W_{\mathrm{out}} \in \mathbb{R}^{C_{\mathrm{total}} \times d_{\mathrm{model}}}, \;
 C_{\mathrm{total}} = \sum_{l=1}^{L} C_l.
-->

The output is partitioned according to each matrix's subcomponent count $C_l$.


**Leaky hard sigmoids**


Theoretically, the causal importance for subcomponent $c$ in matrix $l$ is obtained simply by clamping the outputs of the final transformer layer $z^{(N+1)}_{b,t} $ to the interval$[0,1]$ with a hard sigmoid function:

$$g^l_{b,t,c} = \Gamma(h_{b,t})^l_c=\sigma_{\mathrm{H}}\!\left(z^{(N+1)}_{b,t}\right),
 \quad \sigma_{\mathrm{H}}(z) = \mathrm{clamp}(z, 0, 1),$$
<!-- LaTeX original:
g^l_c(x, t) = \sigma_{\mathrm{h}}\!\left(\Gamma(a(x,t))^l_c\right),
 \quad \sigma_{\mathrm{h}}(z) = \mathrm{clamp}(z, 0, 1),
-->

However, in practice, the flat regions in a hard sigmoid function can lead to dead gradients for inputs below $0$ or above $1$. To avoid this, we use leaky hard sigmoids instead.
Specifically, we use *lower-leaky* hard sigmoids $\sigma_{H,\text{lower}}(x)$ for the causal importance used to create the masks for the actual forward passes for the $\mathcal{L}_{\text{stochastic-recon}}$ and $\mathcal{L}_{\text{stochastic-recon-layerwise}}$ losses, and we use *upper-leaky* hard sigmoids $\sigma_{H,\text{upper}}(x)$ in the importance minimality loss $\mathcal{L}_{\text{importance-minimality}}$ and the frequency minimality loss $\mathcal{L}_{\text{frequency-minimality}}$.

The lower-leaky hard sigmoid $\sigma_{H,\text{lower}}(x)$ has a forward pass identical to a regular hard sigmoid, but below $0$ it uses a straight-through gradient estimator: Gradients pass through for $z \leq 0$ scaled by a leak coefficient $\alpha = 0.01$ when the incoming gradient is negative, preventing subcomponents from becoming permanently deactivated. The upper-leaky hard sigmoid $\sigma_{H,\text{upper}}(x)$ is identical to a regular sigmoid for $z \leq 1$, but has a slope of $0.01$ above $1.0$.

We use a straight-through estimator for the lower-leaky hard sigmoid instead of actually modifying the slope on the forward pass to avoid creating subcomponent masks smaller than zero. We restrict the straight-through estimator to apply only to negative gradients to prevent entries of $\Gamma(h_{b,t})^l_c$ from updating to become ever more negative indefinitely.

This is in contrast to <cite>bushnaq2025spd</cite>, where the lower-leaky hard sigmoid did have an actual slope of $0.01$ below $0$ on the forward pass. We made this change because we discovered that negative masks actually led to instabilities. For example, we found that the spurious subcomponent splitting observed for too-high importance minimality loss coefficients depicted in Figure 8 of that paper largely disappears if the straight-through estimator is used instead.

**Hyperparameters.**

<ref>tab:ci-hyperparams</ref> lists the hyperparameters used for the causal importance function $\Gamma$ in our experiments.

<label id="tab:ci-hyperparams"/>
| **Parameter** | **Value** |
|---|---|
| CI model dimension ($d_{\mathrm{model}}$) | 2048 |
| Transformer layers ($N$) | 8 |
| Attention heads | 16 |
| Head dimension | 128 |
| FFN hidden dimension ($d_{\mathrm{ff}}$) | 8192 |
| Positional encoding | RoPE (base $= 10,000$, max length $= 512$) |
| Attention | Bidirectional (no causal mask) |
| Activation function | Leaky hard sigmoid ($\alpha = 0.01$) |
*Hyperparameters for the causal importance function $\Gamma$*<footnote>Recall that the target model is a 4-layer Llama-style transformer with $d_{\mathrm{model}} = 768$ and
$d_{\mathrm{intermediate}} = 3072$, decomposed across $L = 24$ weight matrices
(6 per layer: `c_fc`, `down_proj`, `q_proj`, `k_proj`,
`v_proj`, `o_proj`), yielding a total of $C_{\mathrm{total}} = 39,936$
subcomponents and an input dimension of $D = 27,648$</footnote>.

### Reconstruction losses

<label id="sec:recon"/>

#### Formal Setup

<label id="sec:vpd_opt-mech-faith-setup"/>

Ablation-based parameter decomposition methods, at their core, instantiate this definition of mechanistic faithfulness by using their causal importance functions (<ref>sec:opt-minimality</ref>) to estimate how ablatable each parameter subcomponent is on a given datapoint. They then actually do an ablation and train the model with ablated parameters to approximate the same output as the unablated model. Crucially, the ablations may be full *or partial*.

Formally, we define ablation masks $m^l_{b,t,c}(r)\in[0,1]$ for each subcomponent at each each batch index $b$ and sequence position $t$. These masks define new weight matrices $W^{\prime l}_{b,t}(r)$ which can take the place of the original model matrices $W^l$:<footnote>For simplicity, we omit the addition of the $\Delta$-component masking term $m^l_{b,t,C+1} \Delta^l_{i,j}$ to this sum.</footnote>

```equation
tex:
  \begin{aligned}
  \htmlClass{hc-maskedparams}{
    W^{\prime l}_{b,t}
  }
  \htmlClass{hc-r}{(r)}
  :=
  \htmlClass{hc-sum-c}{
    \sum^C_{c=1}
    \vec{U^l_c}
      \htmlClass{hc-m}{m^l_{b,t,c}
        \htmlClass{hc-r}{(r)}
      }
    (\vec{V_c^{l}})^\top
  }
  \end{aligned}
tips:
  - hc-maskedparams: The parameter matrix used in place of model matrix l, at this batch and sequence index,
  - hc-r: A tensor that determines the extent of the ablation for each subcomponent at each sequence position on each batch
  - hc-sum-c: The sum of the masked parameter subcomponents
  - hc-m: The mask that ablates the parameters in each subcomponent at each sequence position on each batch
```

<!-- $$
\begin{aligned}
&W^{\prime l}_{b,t,i,j}(r):=\sum^C_{c=1} U^l_{i,c} m^l_{b,t,c}(r) V^l_{j,c} \\
\end{aligned}
$$ -->


<!-- LaTeX original:
\begin{aligned}
&W'^l_{i,j}(x,t,r):=\sum^C_{c=1} U^l_{i,c} m^l_c(x,t,r) V^l_{j,c} \\
\end{aligned}
-->

Crucially, the masks are not the causal importances, $g^l_{b,t,c}$. Instead, the masks are given by

$$m^l_{b,t,c}(r) :=g^l_{b,t,c}+(1-g^l_{b,t,c})r^l_{b,t,c},$$
<!-- LaTeX original:
m^l_c(x,t,r) :=g^l_c(x,t)+(1-g^l_c(x,t))r^l_c(x,t),
-->

where $r^l_{b,t,c} \in [0, 1]$ is called a 'source'. This means that if a subcomponent's causal importance is $1$, the only possible value of its mask is $1$, whereas if the causal importance is $0$, its mask can take any value between $0$ and $1$. The causal importance of the $\Delta$-components $\Delta^l$ is always zero.
<!-- TxDO(Lee): 25 March, 12:20 pm, There's room for a simple figure explaining the above 0/1 interval thing -->
Concretely, when computing the output vector of matrix $l$ at batch index $b$ and sequence position $t$, we replace the original weight matrix $W^l$ with $W^{\prime l}_{b,t}(r)$, which is constructed from the masks at that specific position. This means that during a single forward pass through the network, different linear transformations are applied at each sequence position, determined by which subcomponents are masked on vs. off at that position.
In the idealised setting, we then demand that, for *all possible joint combinations* of sources $r\in {[0,1]}^{L\times B \times T \times C+1}$, the resulting masked weight matrices yield outputs that approximately match those of the original model at every batch index and every output sequence position:


```equation
label: eq:subcomponents
tex:
  \htmlClass{hc-forallr}{ \forall r}
  :
  \htmlClass{hc-ablt-model}{
    f(x_b \vert 
      \htmlClass{hc-ablt-params}{
        W^{\prime 1}_{b}(r),\dots,W^{\prime L}_{b}(r)
        }
      )
    }
  \approx 
  \htmlClass{hc-targ-model}{f(x_b\vert W^1,\dots,W^L)}.
tips:
  - hc-forallr: For every possible value of r
  - hc-ablt-model: The output of the model that uses the parameters with ablations
  - hc-targ-model: The output of the target model
  - hc-ablt-params: The parameters with ablations
```

<!-- $$
\begin{aligned}
&\forall r: f(x_b\vert W^{\prime 1}_{b}(r),\dots,W^{\prime L}_{b}(r))\approx f(x_b\vert W^1,\dots,W^L).
\end{aligned}
$$ -->


<!-- LaTeX original:
\label{eq:subcomponents}
\begin{aligned}
%&W'^l_{i,j}(x,t,r):=\sum^C_{c=1} U^l_{i,c} m^l_c(x,t,r) V^l_{j,c} \\
&\forall r: f(x\vert W'^1(x,t,r),\dots,W'^L(x,t,r))\approx f(x\vert W^1,\dots,W^L).
\end{aligned}
-->

where $f(x_b\vert W^1,\dots,W^L)\in \mathbb{R}^{T}$. This definition of ablatability lies at the heart of how VPD and other ablation-based parameter decomposition methods ensure that the causal importances they provide are mechanistically faithful to the original network.

#### Why is this necessary? Local descriptions must aggregate appropriately

<label id="sec:vpd_recon_motivation"/>

To illustrate why this stricter requirement is necessary, consider the following spurious decomposition which keeps the target model outputs invariant under the joint ablation of all causally unimportant components on every data point in the training dataset and scores very low on $\mathcal{L}_{\text{importance-minimality}}$: 

For every data point $x$, we make up a unique low-rank component $P_x$, and assign it causal importance $1$ on $x$ and $0$ for every other input. We pick the parameters of $P_x$ such that the resulting model exactly matches the final output of the original model: $f(x\vert \theta_x)=f(x\vert \theta)$.<footnote>
To ensure our auxilliary loss $\mathcal{L}_{\text{Delta-L2}}$ is also $0$, we just make up one more component $\theta_{X+1}:=\theta-\sum^X_{x=1} \theta_x$ so that the sum $\sum^{X+1}_{x=1} \theta_x$ equals the target model parameter vector, and assign it causal importance $0$ on every data point.</footnote>
This decomposition would perfectly reconstruct the original model output on every training datapoint, but the resulting components would be spurious and completely unrelated to the mechanistic structure of the target network's learned algorithm. We did not even need to refer to the target model's internals to construct them! They amount to a giant lookup table of the training dataset, and won't generalise to new data points or tell us anything about how the original model actually computed its outputs. 

Requiring that the causally unimportant parameter components can be ablated in any combination rather than just all together excludes counterexamples like this, because it ensures that components do not interfere with the computation on data points they are *not* causally important on. This prevents the decomposition from "splitting up" general computational machinery in the target model into large sets of specialised components that each just memorise a particular input-output pair.

More generally, this stricter requirement ensures that *local descriptions* of the model's behavior on single data points (or small subsets of the dataset) in terms of their causally important parameter components will correctly aggregate into more *global descriptions* of the network's behavior over larger subsets of the dataset in the way we expect: If we explain the network's behavior on two data points $x_1$ and $x_2$ using two different parameter vectors $\sum_{\in S_1} \theta_i, \sum_{\in S_2} \theta_i$, formed from two subsets of the parameter components $S_1, S_2$, a parameter vector formed by the union of both subsets $\sum_{\in S_1 \cup S_2} \theta_i$ will still compute approximately the same output on both datapoints:

$$f(x_1\vert \sum_{\in S_1} \theta_i) \approx f(x_1\vert \sum_{\in S_1 \cup S_2} \theta_i) \quad\text{ AND }\quad f(x_2\vert \sum_{\in S_2} \theta_i) \approx f(x_2\vert \sum_{\in S_1 \cup S_2} \theta_i).$$


#### Stochastic reconstruction losses

<label id="sec:vpd_methods-stoch"/>

We can use an output reconstruction loss to train the masked model's output to approximate the target model's. Unfortunately, to ensure we satisfy <ref>eq:subcomponents</ref>, we would need to do this for *all possible values of* $r\in {[0,1]}^{L\times B\times T \times C+1},$ which is a high dimensional continuous interval, making such a loss impossible to compute exactly. 

However, a key insight of Bushnaq et al. <cite>bushnaq2025spd</cite> was that it is possible to *approximately* minimize reconstruction loss on all values in that interval using a finite number $S$ of uniform random samples $r^{l,(s)}_{b,t,c} \sim \mathcal{U}(0,1)$ for every sequence index $t$ and every batch index $b$. These samples can be used to create stochastic masks $m^l_{b,t,c} \sim \mathcal{U}(g^l_{b,t,c}, 1)$, and minimize reconstruction loss on that finite number of samples.

This leads to the *stochastic reconstruction loss*:

```equation
tex:
  \begin{aligned}
  \mathcal{L}_{\text{stochastic-recon}}
  &=
  \frac{1}{S}
  \sum^{S}_{s=1}
  \frac{1}{B}
  \sum^{B}_{b=1}
  \htmlClass{hc-stoch_rec-divergence}{
    D
    \Big(
      \htmlClass{hc-stoch_rec-stoch_output}{
        f(
          x_b
          \vert
          \htmlClass{hc-stoch_rec-w_stoch}{
            W'_b(
              \htmlClass{hc-stoch_rec-r_stoch_inner}{
                r^{(s)}
              }
            )
          }
        )
      },
      \htmlClass{hc-stoch_rec-target_output}{
        f(
          x_b
          \vert
          \htmlClass{hc-stoch_rec-target_weight}{
            W
          }
        )
      }
    \Big)
  } \\
  \end{aligned}
tips:
  - hc-stoch_rec-divergence: The KL-divergence between the target model and the stochastically masked models.
  - hc-stoch_rec-stoch_output: The decomposed model's output on datapoint x_b
  - hc-stoch_rec-w_stoch: The weight matrix created by stochastically masking parameter components
  - hc-stoch_rec-r_stoch_inner: A sample from a random source
  - hc-stoch_rec-target_output: The target model's output on datapoint x_b
  - hc-stoch_rec-target_weight: The target model's weights
```

<!-- $$\begin{aligned}
\mathcal{L}_{\text{stochastic-recon}}&=\frac{1}{S}\sum^{S}_{s=1}\frac{1}{B}\sum^{B}_{b=1} D \Big( f(x_b\vert W'_b(r^{(s)})),f(x_b\vert W) \Big) \\
\end{aligned}$$ -->

<!-- LaTeX original:
\label{eq:random_recon}
\begin{aligned}
\mathcal{L}_{\text{stochastic-recon}}&=\frac{1}{S}\sum^S_{s=1}D \left( f(x\vert W'(x,t, r^{(s)})),f(x\vert W) \right) \\
\end{aligned}
-->

<!-- Lucius version: -->
<!-- \subsubsection{Optimizing for mechanistic faithfulness: Reconstruction under \textit{stochastic} ablations} -->
<!-- We can use an output reconstruction loss to train the masked model's output to approximate the target model's. Unfortunately, to ensure we satisfy Equation \ref{eq:subcomponents}, we would need to do this for \textit{all possible values of} $r\in {[0,1]}^{C\times L},$ which is a high dimensional continuous interval, making such a loss impossible to compute exactly. However, a key insight of Bushnaq et al. \cite{bushnaq2025spd} was that it is possible to \textit{approximately} minimize reconstruction loss on all values in that interval using a finite number of uniform random samples $r^{l,\text{stoch}}_{c}(x,t) \sim \mathcal{U}(0,1)$ for every sequence position $t$ and every batch index $x$. These samples can be used to create stochastic masks $m^l_c(x,t, g^l_c(x,t)) \sim \mathcal{U}(g^l_c(x,t), 1)$, and minimize reconstruction loss on that finite number of samples.\footnote{In practice, a single sample per batch is used.} This leads to the \textit{stochastic reconstruction loss}: -->

<!-- \begin{equation}\label{eq:random_recon} -->
<!-- \begin{aligned} -->
<!-- \mathcal{L}_{\text{stochastic-recon}}&=D \left( f(x\vert W'(x,t,r^{\text{stoch}})),f(x\vert W) \right) \\ -->
<!-- \end{aligned} -->
<!-- \end{equation} -->

where $D$ is an appropriate divergence measure in the space of model outputs, such as KL-divergence or mean squared error. In practice, we find that using one sample ($S=1$) produces similar training behavior as using more samples. 

In practice, for better convergence, we train by sampling masks for randomly chosen subsets of the model's weight matrices instead of all matrices simultaneously. See the next section for details.

#### Stochastic Subset reconstruction loss

<label id="sec:vpd_subset_recon"/>

<cite>bushnaq2025spd</cite> found that using a reconstruction loss which samples stochastic masks

$$\begin{aligned}
&m^l_{b,t,c}(r^{\text{stoch}}):=g^l_{b,t,c}+\left(1-g^l_{b,t,c}\right)r^{\text{stoch},l}_{b,t,c}\\
&r^{\text{stoch},l}_{b,t,c} \sim \mathcal{U}(0,1)
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
&m^l_c(x,t,r_\text{stoch}):=g^l_c(x,t)+\left(1-g^l_c(x,t)\right)r^l_{\text{stoch},c}(x,t)\\
&r^{l}_{\text{stoch},c}(x,t) \sim \mathcal{U}(0,1)
\end{aligned}
-->

for all target model matrices $l$ simultaneously<footnote>Simply called "stochastic reconstruction loss" in that paper, but here we reserve that term for the formulation that ends up in the training loss.</footnote>

$$\begin{aligned}
\mathcal{L}_{\text{stochastic-recon-all}}&=\frac{1}{B}\sum^B_{b=1} D \left( f\left(x_b\vert {W'}^1\left(r^{\text{stoch}}\right),\dots, {W'}^L\left(r^{\text{stoch}}\right)\right),f\left(x_b\vert W^1,\dots,W^L\right) \right) \\
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
\mathcal{L}_{\text{stochastic-recon-all}}&=D \left( f\left(x\vert {W'}^1\left(x,t,r^{\text{stoch}}\right),\dots, {W'}^L\left(x,t,r^{\text{stoch}}\right)\right),f\left(x\vert W^1,\dots,W^L\right) \right) \\
\end{aligned}
-->

together with a layer-wise stochastic reconstruction loss which samples stochastic masks for one target model matrix at a time

$$\begin{aligned}
\mathcal{L}_{\text{stochastic-recon-layerwise}}=\frac{1}{L}\sum^L_{l=1}\frac{1}{B}\sum^B_{b=1} D \Big( f\left(x_b\vert W^1,\dots,W'^l(r^{\text{stoch}}),\dots,W^L\right),f\left(x_b\vert W^1,\dots,W^L\right) \Big) \\
\end{aligned}$$

<!-- LaTeX original:
\begin{aligned}
\mathcal{L}_{\text{stochastic-recon-layerwise}}=\frac{1}{L}\sum^L_{l=1} D \left( f\left(x\vert W^1,\dots,W'^l(x,t,r^{\text{stoch}}),\dots,W^L\right),f\left(x\vert W^1,\dots,W^L\right) \right)
\end{aligned}
-->

performed better than training either $\mathcal{L}_{\text{stochastic-recon-all}}$ or $\mathcal{L}_{\text{stochastic-recon-layerwise}}$ alone, due to covering a somewhat more structurally diverse set of ablation. However, layer-wise reconstruction loss requires one forward-pass for every matrix in the model we decompose, which is expensive. For VPD training, we unify $\mathcal{L}_{\text{stochastic-recon-all}}$ and layerwise stochastic reconstruction loss $\mathcal{L}_{\text{stochastic-recon-layerwise}}$ into a single stochastic reconstruction loss. For every sequence position and batch index, we independently sample a number $\in\{1,\dots,L\}$, where $L$ is the number of weight matrices in the target model. We draw that many of the target model's weight matrices, sample stochastic masks for only those, and perform a forward pass replacing those matrices with the masked ones. This is no more computationally expensive than $\mathcal{L}_{\text{stochastic-recon-all}}$, and covers more structurally diverse ablations than layer-wise stochastic reconstruction losses, since it includes subsets of single matrices as well as the whole set as special cases.



<!--In <cite>bushnaq2025spd</cite>, we introduced the $\mathcal{L}_{\text{importance-minimality}}$ and $\mathcal{L}_{\text{stochastic-recon}}$ losses to optimize parameter subcomponents to replicate the target model's outputs while using as few parameter subcomponents as possible.-->

Although this reconstruction loss on its own is enough to succeed in many toy settings, our attempts to apply that method at larger scales (such as language models) revealed several pathologies that we missed. We had under-appreciated the importance of reconstruction under worst-case ablation masking which we address in the next section.






#### Adversarial reconstruction loss

<label id="sec:vpd_methods-adv"/>


VPD additionally optimizes for *adversarial ablatability* of parameter subcomponents that are causally unimportant on a datapoint, which is a stricter criterion than *stochastic ablatability*.

In the limit of infinite samples and perfect reconstruction, $\mathcal{L}_{\text{stochastic-recon}}$ loss would perfectly approximate our desired condition from <ref>eq:subcomponents</ref>. But we don't have time to draw infinite samples. And <ref>eq:subcomponents</ref> requires that the masked model approximates the target model well for *all* possible values of $r$, not just on average. Thus, if the reconstruction loss isn't exactly zero, which will essentially always be the case in practice, stochastic sampling can greatly underestimate the worst-case reconstruction error for values of $r$ that are sampled adversarially to maximize reconstruction loss. We found that training without an adversarial sampling scheme produces decompositions for which adversarial sampling can find values of $r$ that have worse-than-random reconstruction loss, which is not permitted under <ref>eq:subcomponents</ref> (See also <ref>fig:adv-vs-no-adv</ref>).

VPD therefore introduces an adversarial loss to help ensure this property: Instead of sampling the sources $r$ randomly, they are sampled by an adversarial optimizer to be as bad as possible.

The optimization objective of the adversarial optimizer is maximizing the reconstruction loss on the masked forward pass:

$$\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}:=\frac{1}{B}\sum^B_{b=1} D \Big( f\left(x_b\vert W'^1(r^{{\text{adv}}}),\dots,W'^L(r^{{\text{adv}}})\right),f(x_b\vert W^1,\dots, W^L) \Big)
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}:=\sum_x D \left( f\left(x\vert W'^1(x,t,r^{{\text{adv}},1}),\dots,W'^L(x,t,r^{{\text{adv}},L})\right),f(x\vert W^1,\dots, W^L) \right)
\end{aligned}
-->

by optimising adversarial sources $r^{{\text{adv}},l}_{b,t,c}$ for the masks $m^l_{b,t,c}(r^{\text{adv}})$:

$$\begin{aligned}
m^l_{b,t,c}(r^{\text{adv}}) &:=g^l_{b,t,c}+(1-g^l_{b,t,c})r^{\text{adv},l}_{b,t,c}\\
W'^l_{b,t,i,j}(r^{{\text{adv}}})&:=\sum^C_{c=1} U^l_{i,c} m^l_{b,t,c}(r^{\text{adv}}) V^l_{j,c}
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
m^l_c(x,t,r^{\text{adv}}) &:=g^l_c(x,t)+(1-g^l_c(x,t))r^{\text{adv},l}_c(x,t)\\
W'^l_{i,j}(x,t,r^{{\text{adv}},l})&:=\sum^C_{c=1} U^l_{i,c} m^l_c(x,t,r^{\text{adv}}) V^l_{j,c}
\end{aligned}
-->
for subcomponent $c$ of target model matrix $l$ on batch index $b$ at sequence position $t$. The optimizer we use is projected gradient ascent <cite>bertsekas1999nonlinear,madry2018towards</cite>, clamping the sources $r^{\text{adv}}_{b,t,c}$ to the interval $[0,1]$ at every update step to ensure that the masks $m^l_c(x,t,r^{\text{adv}})$ stay between $0$ and $1$.
The sources for the $\Delta$-components' masks (see <ref>sec:method-components</ref>) are treated identically to those used for the regular components, i.e. they are also adversarially optimized.


```equation
label: eq:adv_recon
tex:
  \begin{aligned}
  \mathcal{L}_{\text{adversarial-recon}}
  &=
  \htmlClass{hc_adv_rec-root}{
    \htmlClass{hc_adv_rec-max_by}{
      \max_{r^{\text{adv}}}
    }
    \frac{1}{B}
    \sum^{B}_{b=1}
    \htmlClass{hc-adv_rec-divergence}{
      D
      \Big(
        \htmlClass{hc-adv_rec-adv_output}{
          f(
            x_b
            \vert
            \htmlClass{hc-adv_rec-w_adv}{
              W'_b(
                \htmlClass{hc-adv_rec-r_adv_inner}{
                  r^{ \text{adv} }
                }
              )
            }
          ),
        }
        \htmlClass{hc-adv_rec-target_output}{
          f(
            x_b
            \vert
            \htmlClass{hc-adv_rec-target_weight}{
              W
            }
          )
        }
      \Big)
    }
  }
  \end{aligned}
tips:
  - hc_adv_rec-max_by: we optimize adversarial masks to maximise KL divergence over the dataset
  - hc-adv_rec-divergence: The KL-divergence between the target model and the model using adversarially masked parameter components.
  - hc-adv_rec-adv_output: The decomposed model's output on datapoint x_b
  - hc-adv_rec-w_adv: The weight matrix created by adversarially masking parameter components
  - hc-adv_rec-r_adv_inner: the adversarial masks
  - hc-adv_rec-target_output: The target model's output on datapoint x_b
  - hc-adv_rec-target_weight: The target model's weights
```
<!-- $$
\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}&=\textcolor{#E15019}{\max_{r^{\text{adv}}}} \frac{1}{B}\sum^{B,}_{b=1} D \Big( f(x_b\vert W'_b(r^{ \textcolor{#E15019}{ \text{adv} } })),f(x_b\vert W) \Big) \\
\end{aligned}
$$ -->


<!-- markdown original below: just the adv loss by itself -->
<!-- $$
\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}&=\max_{r^{\text{adv}}} \frac{1}{B}\sum^B_{b=1} D \Big( f(x_b\vert W'_b(r^{\text{adv}})),f(x_b\vert W) \Big) \\
\end{aligned}
$$ -->

<!-- LaTeX original:
\label{eq:adv_recon}
\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}&=\max_{r^{\text{adv}}} D \left( f(x\vert W'(x,t,r^{\text{adv}})),f(x\vert W) \right) \\
\end{aligned}
-->

**Complete adversarial robustness seems too strict**

However, if the adversarial sampler were completely unconstrained, it would actually be too strict: Some decompositions that we would intuitively regard as valid would be effectively excluded by it. For example, in many theoretical toy models of circuits in superposition <cite>hänni2024mathematicalmodelscomputationsuperposition, bushnaq2024circuits, linsefors2025circuits</cite> models can contain more circuits than neurons, only some of which are used by the model on any given forward pass. However, the inactive circuits each still contribute some small interference "noise" to the computation. Since this noise is uncorrelated between superposed circuits, its overall size remains small enough that the interference doesn't "break" the computation. We would like to consider these inactive circuits not to be causally important since the model is in some sense not really using them to compute the output. But if we chose the absolute worst-case $r^{\text{adv}}_{c}$ in such a model (which we can do if we have a completely unconstrained adversarial sampler), we could, for example, ablate all inactive circuits which contribute noise with a negative sign, but keep all inactive circuits which contribute noise with a positive sign. This would vastly increase the overall size of the noise and thus change the final output of the model!

In general, we want the adversarial sampler to penalise *systematic* defects in the decomposition, where a particular choice of ablation masks changes the model output on many data points even though it shouldn't. But we do not want the sampler to exploit random noise by finely tuning its choice of ablations to particular data points. This is because in practice, when using the decomposition to understand or edit the target model, we usually care about the behavior of particular component maskings over multiple data points, rather than the behavior of all possible maskings on single data points. 

For example, suppose we wanted to edit the target model for some practical purpose, like erasing some of its knowledge about biology. We could therefore apply a mask to some of the model's components that are causally important in biology contexts, but not other contexts. Ideally, the resulting model should still behave the same way for all inputs on which those components were not causally important.
This mask would be very unlikely to be exactly tuned to random noise in the activations of some other input. And even if it did happen to be so tuned, then this would merely cause the edited model to behave unexpectedly on the input that the mask happened to be tuned to, and thus not be a very effective adversarial mask on other inputs. But if the decomposition was *systematically* defective, we might have a realistic chance of picking a mask that causes the edited model to behave differently than the target model on many inputs not related to biology. This would be an effective adversarial mask that would hurt the model editing more broadly.

Thus, in order to force the adversarial sampler to rely on systematic flaws in the decomposition instead of fine-tuning to individual data points, we restrict it to use the same $r^{\text{adv}}_c$ on all elements in a batch. 

Ideally, we might like to use the same sources for the whole data set, but this would be too computationally expensive in training. In practice, we thus use two different sampling schemes for $r^{\text{adv},l}_{c}$ source schemes for evaluation and training.

**Persistent PGD (PPGD) adversarial reconstruction loss for training:**

For training, we optimize a single set of sources $r^{\text{adv},l}_{b,t,c}$ that persists across batches, with $b$ ranging across the batch index and $t$ across sequence position. On every batch, the adversarial AdamW optimizer performs $n_{\text{adv}}$ update steps on the adversarial sources $r^{\text{adv}}_{b,t,c}$, trying to maximise the adversarial loss $\mathcal{L}_{\text{adversarial-recon}}$ (In this paper, we used $n_{\text{adv}}=3$).

<!-- The adversarial optimizer also supports different scopes for the sources $r^l(x,s)$. \code{per\_batch\_per\_position} uses one unique source for every batch element and sequence position, \code{repeat\_across\_batch} uses the same source $r^l(x)$ for every batch element, but different sources for different sequence positions, and \code{single\_source} uses one unique source $r^l$ for every component. We use \code{per\_batch\_per\_position} in our experiments because it seems to perform best in practice. -->
<!-- \begin{equation}\label{eq:PPGD_recon} -->
<!-- \begin{aligned} -->
<!-- \mathcal{L}_{\text{PPGD recon}}&=\sum_x D \left( f\left(x\vert W'(x,s,r^{l, \text{adv}}\right),f(x\vert W) \right) \\ -->
<!-- \end{aligned} -->
<!-- \end{equation} -->

<!-- The VPD AdamW optimizer performs one update step on the subcomponents and the parameters of the causal importance function by calculating the gradient of the overall VPD loss function components and their causal importance functions. -->
<!-- \begin{equation} -->
<!-- \mathcal{L}_{\text{VPD}}:=\mathcal{L}_{\text{stochastic-recon}}+\mathcal{L}_{\text{adversarial-recon}}+\mathcal{L}_{\text{importance-minimality}}+\mathcal{L}_{\text{frequency-minimality}}+\mathcal{L}_{\text{Delta-L2}}. -->
<!-- \end{equation} -->

**PGD adversarial reconstruction loss for evaluation:**

Continuously updating a single set of persistent adversarial sources is more computationally efficient, but not principled. Hypothetically, the VPD optimizer might trap the adversarial optimizer in some local extremum at some point during training, rendering the adversarial loss useless. Thus, for evaluation, we use a new set of adversarial sources $r^{ \text{adv},l}_c$ for every evaluation batch, but use more adversarial optimization steps per batch $n_{\text{adv}}$ than we do in training.
<!-- Just as with the PPGD loss, the code supports \code{per\_batch\_per\_position}, \code{repeat\_across\_batch} and \code{single\_source} scope. -->


### Frequency minimality loss

<label id="sec:vpd_frequency_penalty"/>

Suppose some rank-1 subcomponent $U_1 \vec{V_1}^\top$ in a model parametrizes two unrelated circuits $A$ and $B$, which are rarely used to compute the model's output at the same batch and sequence position. We would like VPD to break up this subcomponent into two subcomponents, $\vec{U_1} (\vec{V_1})^\top=\vec{U_A} (\vec{V_A})^\top + \vec{U_B} (\vec{V_B})^\top$, with $\vec{U_A} (\vec{V_A})^\top$ containing the weights for circuit $A$, and $\vec{U_B} (\vec{V_B})^\top$ containing the weights for circuit $B$. Our loss $\mathcal{L}_{\text{importance-minimality}}$ will not incentivise this, because either $\vec{U_A} (\vec{V_A})^\top$ or $\vec{U_B} (\vec{V_B})^\top$ will be causally important whenever $\vec{U_1} (\vec{V_1})^\top$ is, so $\sum_{b,t}\vert g_{b,t,1}\vert^p\leq \sum_{b,t}(\vert g_{b,t,A}\vert^p+\vert g_{b,t,B}\vert^p)$. One way to break up subcomponents like $\vec{U_1} (\vec{V_1})^\top$ is introducing an additional loss penalty that is very slightly *superlinear* in causal importance frequency, i.e. penalising a subcomponent that is causally important half of the time more heavily than two subcomponents that are each active a quarter of the time. 

This leaves the question of what precise functional form this superlinear penalty should take. We ultimately opted for a term that grows approximately as $\sum^L_{l=1}\sum^C_{c=1}f^l_c \log_2(f^l_c)$ with causal importance frequency $f^l_c:=\frac{1}{BT}\sum^B_{b=1}\sum^T_{t=1}\vert g^l_{b,t,c}\vert^0$. This was largely motivated by empirical iteration, though <ref>app:frequency_penalty_motivation</ref> provides some theoretical motivation for the log scaling based on minimising mechanistic description length per data point: The effective description length of subcomponents in bits (weakly) grows with $\log_2(f^l_c)$, because subcomponents that activate more frequently effectively need to be specified to higher precision to mantain good output reconstruction.

The normalisation $\frac{1}{BT}$ inside the $\log_2$ argument can be absorbed into the importance minimality loss term via the relation $\log_2(f^l_c)=\log_2(\sum^B_{b=1}\sum^T_{t=1}\vert g^l_{b,t,c}\vert^0)-\log_2(BT)$. Adding a $1.0$ inside the $\log_2$ for numeric stability and using $L_p$ norm in place of $L_0$ then yields


$$\begin{aligned}
\mathcal{L}_{\text{frequency-minimality}}=\frac{1}{BT}\sum^L_{l=1}\sum^B_{b'=1}\sum^T_{t'=1}\sum^C_{c=1}\vert g^l_{b',t',c}\vert^p \log_2(1+\sum^B_{b=1}\sum^T_{t=1} \vert g^l_{b,t,c}\vert^p)\,.
\end{aligned}$$


### p-annealing

The $L^p$ quasi-norm in the importance minimality loss $\mathcal{L}_{\text{importance-minimality}}$ and frequency minimality loss $\mathcal{L}_{\text{frequency-minimality}}$ (<ref>eq:minimal</ref> and <ref>eq:freq_minimality</ref>) serves as a
smooth surrogate for the $L_0$ ‘norm’, with smaller $p$ yielding a tighter approximation.
However, <cite>bushnaq2025spd</cite> found that optimization is substantially easier at larger $p$
values like $p = 2$. We therefore linearly anneal $p$ over the
course of training, starting from the easy-to-optimize $p_0 = 2.0$ and decreasing to
$p_{\mathrm{final}} = 0.4$:

$$p(t) = p_0 + (p_{\mathrm{final}} - p_0) \cdot \frac{t}{t_{\max}},$$
<!-- LaTeX original:
p(t) = p_0 + (p_{\mathrm{final}} - p_0) \cdot \frac{t}{t_{\max}},
-->

where $t$ is the current training step and $t_{\max}$ is the total number of steps.
In our experiments, annealing begins at the start of training and proceeds linearly
over the full run ($t \in [0, t_{\max}]$).




## Appendix

<label id="sec:app:appendix"/>

### Clustering Subcomponents into Components {toc: Clustering Subcomponents}

<label id="app:clustering"/>

VPD decomposes each weight matrix $W_l$ into a sum of rank-one *subcomponents*: $W_l \approx \sum_{c} \vec{U^l_c} (\vec{V_c^{l}})^\top$. While each subcomponent only spans a single weight matrix, a full *parameter component* could span the entire parameter space, potentially involving subcomponents from multiple weight matrices. We therefore need a method to identify which subcomponents across different weight matrices should be grouped together into coherent parameter components.

#### Coactivation-Driven Clustering

We observe that subcomponents that participate in the same computation should activate together on the same datapoints. If subcomponent $c$ from layer $l$ and subcomponent $c'$ from layer $l'$ consistently have high causal importance values on the same inputs, we suppose that they are likely implementing related computations and should be grouped into the same parameter component.

Let $g^l_{b,t,c} \in [0, 1]$ denote the causal importance of subcomponent $c$ in layer $l$ at sequence position $t$ on batch index $b$. Given a dataset $\mathcal{D} = \{x_1, \ldots, x_N\}$, we compute a *coactivation matrix* that measures how often pairs of subcomponents activate together:

$$\text{CoAct}_{i,j}
 = \sum_{x \in \mathcal{D}}
 \mathbf{1}[ g_{b,t,i}(x) > \tau ]
 \cdot \mathbf{1}[ g_{b,t,j}(x) > \tau ]$$
<!-- LaTeX original:
\text{CoAct}_{i,j}
 = \sum_{x \in \mathcal{D}}
 \mathbf{1}[ g_i(x) > \tau ]
 \cdot \mathbf{1}[ g_j(x) > \tau ]
-->

where $\tau$ is an activation threshold (we use $\tau = 0.01$ by default) and indices $i, j$ enumerate all subcomponents across all layers. The diagonal entry $\text{CoAct}_{i,i}$ gives the total activation count for subcomponent $i$.

#### Minimum Description Length Clustering

We frame the clustering problem using the *Minimum Description Length* (MDL) principle, which states that the shortest description of the data is the best one. The goal is to find a grouping of subcomponents that minimizes the total cost of describing both the grouping structure and the activation patterns of the grouped components. Since the number of possible groupings grows according to Stirling numbers of the second kind, enumerating all partitions is infeasible. Instead, we use a stochastic merging approach guided by the MDL cost.

Consider a partition of $n$ subcomponents into $k$ groups $\{P_1, \ldots, P_k\}$. For each group $P_i$, let:

- $s_i = \text{CoAct}_{i,i}$ denote the activation count (diagonal of coactivation matrix after grouping)
- $r(P_i) = |P_i|$ denote the *rank* of the group (number of subcomponents it contains)

The MDL cost for the current grouping is:

$$\mathcal{L}_{\text{MDL}}
 = \sum_{i=1}^{k} s_i \left( \log_2(k)
 + \alpha \cdot r(P_i) \right)$$
<!-- LaTeX original:
\mathcal{L}_{\text{MDL}}
 = \sum_{i=1}^{k} s_i \left( \log_2(k)
 + \alpha \cdot r(P_i) \right)
-->

This cost has an intuitive interpretation: each time a group activates, we must encode (1) which group it is ($\log_2(k)$ bits) and (2) the rank-one matrices comprising that group ($\alpha \cdot r(P_i)$ bits, where $\alpha$ controls the penalty for group complexity). Intuitively, $\alpha$ controls how much we care about the average description length of the matrices we need to inspect to understand how the target model computes its output on any one data point, which matters to us because we assume that causal graphs with longer description lengths tend to be harder for us to understand. On the other hand, the $\log_2(k)$ quantifies the description length of the sets of components involved in calculating the model output for each input across a whole dataset. We care about this description length because we assume that if the same set of components is used on different data points, it will be easier for us to unify and generalise our separate explantions of the model's behavior on many different inputs into a single explanation of the model's behavior on all those inputs.

**Choosing alpha**

As an intuition pump for choosing the $\alpha$ hyperparameter in practice, consider two rank-1 components $P^1, P^2$ with causal importances $s_1, s_2$ that are exactly zero on all data points, where component $P_2$ is causally important with some probability $\mathrm{co}(P_2\mid P_1):=\Pr(P_2\text{ important}\mid P_1\text{ important})$ conditional on $P_1$ being causally important. If the total dictionry size is large enough that we can approximate $\log_2(k-1)\approx \log_2(k)$, the mdl loss will be lowered by merging these two components into one if 
$$\alpha
< \frac{\mathrm{co}(P_2\mid P_1)}{1-\mathrm{co}(P_2\mid P_1)}\cdot \frac{\log_2(k)}{2}.$$

#### Stochastic Hierarchical Merging

We use a stochastic hierarchical clustering algorithm that starts with each subcomponent in its own group and iteratively merges pairs to reduce the MDL cost. At each iteration, we compute the *merge cost* for combining groups $P_i$ and $P_j$. Let $s_{i,j}$ denote the activation count of the merged group (computed as the `OR` of the activation indicators), and let $s_\Sigma = \sum_i s_i$ be the total activation count. The change in MDL cost from merging is:

$$
\begin{aligned}
\Delta\mathcal{L}(P_i, P_j)
&= \underbrace{
(s_\Sigma - s_i - s_j) \log_2 \frac{k-1}{k}
}_{\text{dictionary reduction}} \\
&+ \underbrace{
s_{i,j} \log_2(k-1) - s_i \log_2 (k) - s_j \log_2 (k)
}_{\text{index encoding}} \\
&+ \underbrace{
\alpha \left( s_{i,j} \cdot r(P_{i,j}) - s_i \cdot r(P_i) - s_j \cdot r(P_j) \right)
}_{\text{rank penalty}}
\end{aligned}
$$
<!-- LaTeX original:
\Delta\mathcal{L}(P_i, P_j)
 &= \underbrace{
 (s_\Sigma - s_i - s_j) \log_2 \frac{k-1}{k}
 }_{\text{dictionary reduction}} \\
 &+ \underbrace{
 s_{i,j} \log_2(k-1) - s_i \log_2 (k) - s_j \log_2 (k)
 }_{\text{index encoding}} \\
 &+ \underbrace{
 \alpha \left( s_{i,j} \cdot r(P_{i,j}) - s_i \cdot r(P_i) - s_j \cdot r(P_j) \right)
 }_{\text{rank penalty}}
-->

where $r(P_{i,j})$ is the rank of the merged group. For simplicity, we approximate that $r(P_{i,j}) \approx r(P_i) + r(P_j)$.

Naively, one might greedily select the pair $(i^*, j^*) = \arg\min_{i < j} \Delta\mathcal{L}(P_i, P_j)$ and merge them, but this risks getting stuck in local minima. To allow for more exploration of the space of possible clusterings, we use stochastic selection: instead of always choosing the minimum-cost pair, we sample from all pairs using a probability distribution that exponentially decays with higher cost. 
Specifically, we rank all candidate merge pairs by their cost $\Delta\mathcal{L}$ in ascending order and assign each pair a probability that decays exponentially in its rank:                            
$$P \propto \exp(-\gamma \cdot J), \quad J = 0, 1, \ldots, \tbinom{k}{2} - 1$$ 

where $J = 0$ corresponds to the lowest-cost pair and $\gamma > 0$ is a decay rate controlling exploration. Setting $\gamma \to \infty$ recovers greedy selection, while $\gamma \to 0$ gives uniform sampling. We sample efficiently via the inverse CDF: letting $N = \binom{k}{2}$ be the number of candidate pairs and $u \sim \text{Uniform}(0,1)$, the sampled rank is                            
$$J = \left\lfloor \frac{-\log\bigl(1 - u(1 - e^{-\gamma N})\bigr)}{\gamma} \right\rfloor.$$

where $\lfloor \dots \rfloor$ is the floor function rounding down to the nearest integer. In our experiments, we use $\gamma = 0.2$, which concentrates most probability mass on the top few candidates while maintaining meaningful probability on roughly the five lowest-cost merges. This stochastic selection allows the algorithm to escape local minima that greedy merging would get trapped in, while still strongly preferring merges that reduce the MDL cost.


<!-- Consider two rank-1 components $P_i, P_j$ with importances $s_i, s_j$, merged into a single component $P_{ij}$ with importance $s_{ij}$ and rank $r(P_{ij})=2$. We assume that $s_i(x), s_j(x)\in\{0,1\}$, which can be achieved by rounding causal importance values to $0$ or $1$ depending on whether they are below or above some cutoff. Then, pointwise, we have

$$s_{ij}(x) = s_i(x)\lor s_j(x) = s_i(x)+s_j(x)-s_i(x)s_j(x).$$
LaTeX original:
s_{ij}(x) = s_i(x)\lor s_j(x) = s_i(x)+s_j(x)-s_i(x)s_j(x).


After merging, the number of components becomes $k' = k-1$.
Let $s_\Sigma := \sum_{\ell=1}^k s_\ell$.

The change in MDL loss before and after the merge is then

<label id="eq:delta_simplified"/>
$$
\begin{aligned}
&\Delta \mathcal{L}_{\text{MDL}}(P_i,P_j) \\
&= (s_\Sigma - s_i - s_j)\log_2\!\Bigl(\tfrac{k-1}{k}\Bigr)
+ s_{ij}\log_2(k-1) - (s_i+s_j)\log_2(k)
+ \alpha(2s_{ij} - s_i - s_j).
\end{aligned}
$$
LaTeX original:
\label{eq:delta_simplified}
 &\Delta \MDL(P_i,P_j) \\
 &= (s_\Sigma - s_i - s_j)\log_2\!\Bigl(\tfrac{k-1}{k}\Bigr)
 + s_{ij}\log_2(k-1) - (s_i+s_j)\log_2(k)
 + \alpha(2s_{ij} - s_i - s_j).


If we additionally approximate $\log_2(k-1)\approx \log_2(k)$ and neglect the small dictionary term
$\log_2\!\bigl(\tfrac{k-1}{k}\bigr)$,
this simplifies pointwise to

$$\Delta \mathcal{L}_{\text{MDL}}(x) \approx
 \alpha\bigl(s_i(x)+s_j(x)-2s_i(x)s_j(x)\bigr)
 -\log_2(k)\, s_i(x)s_j(x).$$
LaTeX original:
\Delta \MDL(x) \approx
 \alpha\bigl(s_i(x)+s_j(x)-2s_i(x)s_j(x)\bigr)
 -\log_2(k)\, s_i(x)s_j(x).


Let the dataset have $X$ examples and write empirical averages as
$\mathbb{E}[\cdot] := \frac{1}{X}\sum_{x=1}^X(\cdot)$.
Averaging <ref>eq:delta_simplified</ref> then gives

<label id="eq:avg_delta"/>
$$\mathbb{E}[\Delta \mathcal{L}_{\text{MDL}}] \approx
 \alpha\bigl(\mathbb{E}[s_1]+\mathbb{E}[s_2]\bigr)
 -(2\alpha+\log_2 k)\,\mathbb{E}[s_1 s_2].$$
LaTeX original:
\mathbb{E}[\Delta \MDL] \approx
 \alpha\bigl(\mathbb{E}[s_1]+\mathbb{E}[s_2]\bigr)
 -(2\alpha+\log_2 k)\,\mathbb{E}[s_1 s_2].
 \label{eq:avg_delta}


If $\mathbb{E}[s_1]=\mathbb{E}[s_2]$, then

$$\mathbb{E}[\Delta \mathcal{L}_{\text{MDL}}] \approx
 2\alpha\,\mathbb{E}[s_1]
 -(2\alpha+\log_2 k)\,\mathbb{E}[s_1 s_2].$$
LaTeX original:
\mathbb{E}[\Delta \MDL] \approx
 2\alpha\,\mathbb{E}[s_1]
 -(2\alpha+\log_2 k)\,\mathbb{E}[s_1 s_2].


For binary indicators, we can write

<label id="eq:co_def"/>
$$\mathbb{E}[s_1 s_2]
 = \Pr(P_2\text{ important}\mid P_1\text{ important})\;\mathbb{E}[s_1].$$
LaTeX original:
\mathbb{E}[s_1 s_2]
 = \Pr(P_2\text{ important}\mid P_1\text{ important})\;\mathbb{E}[s_1].
 \label{eq:co_def}


Denoting $\mathrm{co}(P_2\mid P_1):=\Pr(P_2\text{ important}\mid P_1\text{ important})$,
plugging <ref>eq:co_def</ref> into <ref>eq:avg_delta</ref> and canceling $\mathbb{E}[s_1]$ (assuming $\mathbb{E}[s_1]>0$) yields the threshold condition

$$
\begin{aligned}
0 &= 2\alpha - \mathrm{co}(P_2\mid P_1)\,(2\alpha+\log_2 k),
\end{aligned}
$$
 LaTeX original:
0 &= 2\alpha - \mathrm{co}(P_2\mid P_1)\,(2\alpha+\log_2 k),


so

$$\alpha
= \frac{\mathrm{co}(P_2\mid P_1)}{1-\mathrm{co}(P_2\mid P_1)}\cdot \frac{\log_2(k)}{2}.$$
LaTeX original:
\alpha
= \frac{\mathrm{co}(P_2\mid P_1)}{1-\mathrm{co}(P_2\mid P_1)}\cdot \frac{\log_2(k)}{2}.
-->

<!-- \subsection{Ensemble Clustering and Stability Analysis} -->

<!-- A single clustering run may be sensitive to the specific dataset batch used to compute coactivations. To assess clustering stability, we run an \textit{ensemble} of clustering experiments with different random seeds (which determine which data samples are used for computing coactivations). -->

<!-- To compare clusterings across ensemble runs, we need a distance metric that is invariant to the arbitrary labeling of groups. We use the \textit{permutation-invariant Hamming distance}: given two label vectors $\mathbf{a}, \mathbf{b} \in \{1, \ldots, k\}^n$ (assigning each of $n$ subcomponents to one of $k$ groups), we find the optimal permutation $\pi^*$ of group labels that maximizes agreement: -->

<!-- \begin{equation} -->
<!-- d_{\text{PIH}}(\mathbf{a}, \mathbf{b}) -->
<!-- = n - \max_{\pi} \sum_{c=1}^{n} \mathbf{1}[a_c = \pi(b_c)] -->
<!-- \end{equation} -->

<!-- The optimal permutation is found efficiently using the \txdo{txdo} on the confusion matrix between label assignments. Low distances across ensemble runs indicate stable, reproducible groupings that likely reflect genuine computational structure in the network. -->






<!-- ## Further analysis of Static Interaction Strength



<!-- Nice to have: txdo: In theory, it should be the case that a weighted sum of the active component interactions (weighted by the strength of the activations of both subcomponents) should have a close correspondence to the logits of the input at that head. We demonstrate that this is in fact the case, helping to give us some assurance that decomposing attention into AttentionContributions is a useful approach for decomposing attention -->


<!-- We can see by comparing the sum of the Standardized Static Interaction Strength between all pairs (<ref>fig:attn_contrib_grid</ref> – black lines) that it closely corresponds to the shape of the model's actual average attention logits on real or random tokens (<ref>fig:attn_patterns_layer1</ref>). This is somewhat striking because the (Standardized)AttentionContribution is a purely static measure of interaction strength between components, yet it lets us somewhat accurately predict how the head behaves on actual data on average.  -->

<!-- Graveyard: This supports the idea that component interactions are a reasonable way to decompose how attention at this layer actually works in the average case. -->

<!-- <figure>
<label id="fig:attn_patterns_layer1"/>
<img src="figures/layer1_qk_contribution_and_logits.png">
<figcaption>The sum of the parameter component interactions (a static measure) tends to roughly capture the shape of the average logits on data.</figcaption>
</figure>
 --> -->



### A training recipe for VPD

<label id="app:recipe"/>


<!-- Training recipe: Which metrics matter and what (relative/absolute) values they should have -->

In this section, we offer practical guidance for applying VPD to other language models, based on our experience training with the model studied in this paper, as well as a range of other toy models. See <ref>app:training-details</ref> for the hyperparmeters used in the decomposition studied in this paper.

**Evaluation metrics.**

To assess whether a VPD decomposition has converged to a satisfactory solution, we recommend tracking the following primary metrics:

1. **PGD reconstruction loss** (adversarial masks, freshly initialized at each step): The most important metric. This evaluates reconstruction quality under adversarially chosen masks optimized independently for each batch. The setting we want is `per_batch_per_position`, see <ref>sec:vpd_methods-adv</ref> for why. This is stricter than the persistent adversarial loss used during training and is our primary indicator of mechanistic faithfulness. For deeper models, more adversarial steps may be needed. As a rough heuristic, we keep $n_{\text{adv}} \cdot \text{lr}_{\text{adv}} \approx 2$; if increasing the number of steps, decrease the learning rate proportionally so the adversarial optimizer can tune more precisely. For discussion on how much adversarial optimization exactly our causal importances should be robust to, see <ref>sec:discussion</ref>.
2. **$L_0$ per data point**: The average number of subcomponents with nonzero causal importance on a data point. This should be tracked relative to the rank of the original weight matrices. For a transformer, MLP matrices typically have rank $d_{\text{resid}}$; the $L_0$ should be significantly smaller than this for the decomposition to be providing a useful simplification. Note that $L_0$ typically starts high and decreases steadily over training due to $p$-annealing (see below), so early in training the importance minimality loss value is a better predictor of what the final $L_0$ will be.

Additionally, we often monitor **Stochastic reconstruction loss**, because it indicates performance under the average permitted masking as opposed to worst-case maskings, **unmasked reconstruction loss** (all masks set to $1.0$, excluding the $\Delta$-components), because it indicates to what extent the sum of all subcomponents is identical to the target model even without the $\Delta$-components and **CI-masked reconstruction loss** (using the causal importance values directly as masks) as well as **Rounded CI-masked reconstruction loss** (as CI-masked but all causal importance greater than zero are rounded to $1.0$) because they indicate performance when keeping exactly those subcomponents deemed causally important. Note though that the latter two are only useful indicators because VPD does not directly optimize for them: It would be (and in practice is) trivial to achieve almost perfect reconstruction for these two maskings if we included them in the training loss. But this would not indicate that our decomposition was actually capturing more of the target model's computation, because these metrics are not robust to "cheating" in the way the adversarial, and to a lesser extent stochastic reconstruction losses are.

**Training Loss terms.**

VPD training uses the following loss terms, each of which requires its own loss coefficient. We discuss considerations for tuning these below.

1. **Adversarial reconstruction loss** ($\mathcal{L}_{\text{PPGD recon}}$, coefficient $0.5$): This is the persistent PGD loss described in <ref>sec:vpd_methods-adv</ref>. Making the adversarial optimizer cheap yet effective is nontrivial. The adversarial learning rate usually needs to be tuned and depends on the regular learning rate. For the other hyperparameters of the adversarial optimizer, we recommend using the defaults described in <ref>app:training-details</ref>: an Adam optimizer with $\beta_1 = 0.5$, $\beta_2 = 0.99$, constant learning rate with short warmup, per-batch-per-position source scope, and updating the sources $n_{\text{adv}}=3$ times for each outer step (in our implementation, we do two inner "warmup" steps and then apply the outer loss step which also updates the sources). For smaller models, fewer adversarial steps per training step may suffice; for larger, especially deeper, models may need more steps (and a correspondingly lower adversarial learning rate). We usually keep this loss coefficient fixed to $0.5$, setting the scale for the other losses.
2. **Stochastic reconstruction loss** ($\mathcal{L}_{\text{stochastic-recon}}$, coefficient $0.5$): This loss primarily prevents the optimization from stalling early in training, and secondarily prevents it from over-focusing on worst-case ablations at the expense of average-case reconstruction quality. We keep the coefficients for the two reconstruction losses equal and normalized to $\frac{1}{2}$ each. We usually keep this loss coefficient fixed to $0.5$, setting the scale for the other losses.
3. **Importance minimality loss** ($\mathcal{L}_{\text{importance-minimality}}$): This is typically one of the most sensitive hyperparameters and often requires tuning. The $p$-norm exponent is annealed linearly from $p_0 = 2.0$ to $p_{\text{final}} = 0.4$ over the full training run. We recommend keeping this annealing schedule fixed and tuning the coefficient instead. Setting the coefficient too high leads to collapsed decompositions with poor reconstruction; too low leads to decompositions where too many subcomponents are simultaneously active.
4. **Frequency minimality loss** ($\mathcal{L}_{\text{frequency-minimality}}$): The coefficient for this term also requires some tuning, but interacts with the importance minimality coefficient: increasing the frequency minimality loss coefficient effectively increases sparsity pressure, so it may be necessary to lower the importance minimality loss coefficient to compensate. As a starting point, we suggest setting the frequency minimality loss coefficient at roughly $0.5\times$ the importance minimality coefficient, unless problems are observed. Too low a coefficient tends to produce fewer, overly polysemantic components.
5. **$\Delta$-component L2 penalty** ($\mathcal{L}_{\text{Delta-L2}}$): This penalizes the MSE between the sum of subcomponents and each target weight matrix. In practice, this coefficient is not very sensitive. We recommend increasing it by factors of $10$ from a conservative starting point until the unmasked reconstruction loss becomes negligibly small. It is safe to overshoot the coefficient considerably, though making it excessively large can still impair optimization.

**Subcomponent count $C$.**
 The number of rank-one subcomponents per weight matrix is not extremely sensitive. It should be set large enough for the optimization to capture all the subcomponents that are present. If unsure, we recommend erring on the side of too many subcomponents, then inspecting the spectrum of log mean causal importances (averaged over a batch) at the end of an exploratory run. There is typically a sharp cutoff in this spectrum separating "alive" from "dead" subcomponents, which reveals how many subcomponents are actually in use<footnote>We've found the log mean causal importance spectra much more valuable as a measure of the number of "dead subcomponents" compared to counting the number of datapoints on which a subcomponent fires at all. There are often some very small firings that aren't meaningful, making choosing a cutoff difficult.</footnote>. The optimization tends to work best when $C$ is larger than needed, but not excessively so—roughly within a factor of $2$ of the true number of subcomponents appears to work well.

**Causal importance function**
 For decomposing transformer models, we recommend using `global_shared_transformer` as the causal importance function. This is itself a transformer model, which receives the concatenated hidden activations of the target model as input, and produces causal importances for all subcomponents as output. We typically choose the depth of this transformer to be within $\frac{1}{2}-2$ times the depth of the target model, though we have not investigated this hyperparameter as much as some others. We choose the residual stream to be wider than that of the target model since it needs to accommodate all of its hidden activations. For this paper, we used $2048$ compared to $768$ for the target model. As is somewhat standard, we usually choose the MLP width to be approximately four times the width of the residual stream.

**Summary**
 Applying the method to a new model usually requires adjusting

1. The importance minimality loss coefficient.
2. The learning rate
3. The adversarial learning rate
4. The frequency minimality loss coefficient
5. The number of subcomponents $C$
6. The Delta L2 penalty loss coefficient.

In our experience, the first three typically require the most extensive tuning. For larger models, the size of the model used for the causal importance function will likely need to be increased as well. The number of adversarial steps and the adversarial learning rate may also require adjustment. 


### End-to-end transcoders

<label id="app:vpd-sparsity-acc-tradeoff"/>

In <ref>sec:decomp-model-behav-sim</ref>, we showed that VPD Pareto-dominates MSE-trained PLTs and CLTs under all three sparsity measures (<ref>fig:pareto-mse</ref>). However, that advantage may partially reflect a difference in training signal, since VPD optimizes end-to-end on the output distribution while the transcoders optimize layer-wise MSE. Here we control for this by training all activation-based methods with the same end-to-end KL-divergence objective as VPD.

**Training and evaluation protocols**

<label id="app:mode-mismatch"/>

When we replace all MLP layers simultaneously, there is an important design choice: should each layer's encoder see the *clean* residual stream (as computed by the original model) or the *modified* residual stream (which includes reconstruction errors from earlier layers)? We call these the ***clean-input*** and ***error-propagating*** evaluation protocols, respectively. A third option, ***single-layer***, replaces only one MLP at a time, with all other layers left unmodified. For a perfectly faithful reconstruction — one that exactly replicates each MLP's computation — these three protocols would produce similar results.

We train separate sweeps of BatchTopK PLTs and CLTs ($k \in \{8, 16, 32\}$) in clean-input and error-propagating mode, as well as single-layer-trained PLTs. All use KL divergence on the output logits as the training loss, matching VPD. Each model is then evaluated under all three protocols. <ref>fig:pareto-e2e</ref> shows the results.

<figure class="wide">
<label id="fig:pareto-e2e"/>
<img src="figures/pareto_e2e_v4.png">
<figcaption>CE degradation vs. L0 (active features per module) for end-to-end KL-trained methods under three evaluation protocols. **(a)** Error-propagating: each encoder sees the modified residual stream. **(b)** Clean-input: each encoder sees the clean residual stream. **(c)** Single-layer replacement, averaged over layers. PLTs (blue) and CLTs (orange) perform well in their training mode but degrade by 5-20x in the mismatched mode. VPD (purple markers) is relatively stable across all three protocols. Linestyle indicates training mode: solid = error-propagating, dashed = clean-input, dotted = single-layer.</figcaption>
</figure>

**Activation-based methods are brittle to mode mismatch.**

The activation-based methods exhibit severe brittleness to evaluation mode mismatch. In the matched setting, error-propagating-trained PLTs achieve CE degradation as low as $\delta = 0.32$, and clean-input-trained PLTs reach $\delta = 0.18$ at $k=32$ (<ref>fig:pareto-e2e</ref>b). But when evaluated in the *mismatched* setting, these same models degrade catastrophically: clean-input-trained models evaluated in error-propagating mode suffer $\delta \approx 2.9$—$3.5$, roughly an order of magnitude worse. The pattern is symmetric: error-propagating-trained models fail in clean-input evaluation ($\delta \approx 1.6$—$2.2$). CLTs exhibit the same pattern. The gap between matched and mismatched performance is a factor of $3$—$20\times$.

This brittleness reveals that e.g. a PLT trained in error-propagating mode does not simply learn to approximate each MLP's input-output function. Instead, it learns a replacement model that *jointly* accounts for both the MLP's true computation and the systematic reconstruction errors introduced by the PLTs in earlier layers. This is a compensatory strategy rather than a mechanistically faithful approximation of the original target model.

Single-layer-trained PLTs, which each see only the clean residual stream for their own layer, are the most robust of the activation-based methods, and perform best in the single-layer replacement setting ($\delta \approx 0.13$—$0.19$). However, when all four single-layer-trained PLTs are inserted simultaneously, they still exhibit meaningful degradation ($\delta \approx 0.56$—$0.99$), because each was trained in isolation and cannot account for reconstruction errors accumulating from other layers.

**VPD is stable across protocols.**

VPD's CE degradation, by contrast, is relatively consistent across all three evaluation protocols. At CI$>$0, VPD achieves $\delta \approx 0.32$–$0.42$ regardless of whether it is evaluated in error-propagating, clean-input, or single-layer mode. This arises because VPD's stochastic and adversarial masking during training already exposes the decomposition to a rich diversity of partial ablation patterns: on each training step, a random subset of subcomponents across random subsets of weight matrices are partially masked, which naturally covers patterns resembling both error-propagating and clean-input replacement as special cases. More fundamentally, VPD's subcomponents sum to the original weight matrices, and the masked forward pass uses the same architecture and nonlinearities as the target model. A VPD reconstruction is therefore not a different function approximating the MLP, but rather a subset of the MLP's computations.

That said, VPD does not achieve the lowest CE degradation in every individual setting. In matched-mode evaluation, the best activation-based models outperform it (e.g., clean-input PLTs at $k=16$ reach $\delta \approx 0.23$ vs. VPD's $\delta \approx 0.42$). We view this as the expected cost of faithfulness: a model specifically optimized to compensate for a particular error pattern will naturally outperform one that has not learned such compensation.





### Confirming feature splitting in PLTs and CLTs geometrically {toc: Feature splitting in PLTs and CLTs}

<label id="app:confirming-feature-splitting"/>


To confirm that the PLTs and CLTs are indeed splitting features rather than discovering genuinely new ones, we match features between models of different sizes. For each pair of models, we count what fraction of alive components in one model have more than one match among the alive components of the other model, averaged across layers. We match components by calculating the cosine similarity between their output vectors (decoder vector for PLT/CLT; down-projection $\vec{U}$ vector for VPD) and consider a cosine similarity $> 0.5$ a match. Results are qualitatively stable across cosine similarity thresholds in $[0.3, 0.7]$. A component with multiple matches in a target model is evidence that the target model has split what the source model represents as a single feature.

<figure class="fig-simplicity">
<label id="fig:splitting-heatmap"/>
<img src="figures/split_heatmap.png" alt="Cross-model feature splitting heatmap">
<figcaption>Cross-model component splitting (cosine $> 0.5$). Each cell shows the percentage of alive components in the source model (row) that have more than one match in the target model (column). VPD models show low splitting both within and across models. PLTs and CLTs show high mutual splitting, indicating substantial redundancy among their learned features.</figcaption>
</figure>

The heatmap confirms that the proportion of features that have multiple matches in a version with more components is higher in PLTs and CLTs. For example, 57.0% of the components in 4k PLT have more than one match in the 32k PLT. On the other hand, only 2.7% of the components in the 0.5x VPD model have more than one match in 8x VPD model. 

### Geometric consistency across seeds {toc: Consistency across seeds}

<label id="app:seed-stability"/>


In mechanistic interpretability, it is common to look for the 'mechanisms' or 'features' that a network uses in its computations. There's an implication here: That there is a fixed, ground truth set of objects that we're looking for ("*the* mechanisms"). How true is this? And how would we measure how close we are to finding the right objects? 

One approach is to run a decomposition method with different random seeds or using different hyperparameters. If the approaches converge to the same results despite these differences, this is suggestive that they converged to the 'right' set of objects. 

Previous work has used mean max cosine similarity (MMCS) to measure this similarity quantitatively <cite>Sharkey_Braun_Millidge_2022</cite>. Suppose perform two decompositions using the same method, but with different random seeds. Given these two sets of transcoder latents or a subcomponents, we calculate the cosine similarity between the objects in each set, and find the most similar for each, and the take the average cosine similarity between those maximally similar pairs. High MMCS means that decompositions are similar across seeds. 

Since VPD is trained using an end-to-end (e2e) loss we compare it with transcoders trained with an e2e loss (<ref>tab:seed-mmcs</ref>). We find that the MMCS of the transcoder latents is similar or slightly worse than the MMCS of VPD U and V vectors. But PLTs and CLTs are usually not trained with an e2e loss; they are usually trained to reconstruct activations at each layer (i.e. a 'local MSE' loss). VPD does not typically train with a hidden activation reconstruction loss; if it reconstructed hidden activations perfectly, it would be constructing activations that are not relevant for performance and merely correspond to 'superposition noise'. Despite not training on hidden activation reconstruction loss, the constellation of other loss functions results in a hidden activation reconstruction loss that is similar, albeit slightly higher, than if we do minimize it directly (Stochastic forward pass hidden activation MSE: 0.33 vs. 0.41). When transcoders are trained using their typical training loss (local MSE), their MMCS are much better than VPD.
<!-- hidden act aux loss run: https://wandb.ai/goodfire/spd/runs/s-aa4fec0a?nw=nwuserdanbraun vs Jose --> 

<label id="tab:seed-mmcs"/>
<table>
<tr><th>Method</th><th>Cross-seed Mean Max Cos Sim</th></tr>
<tr><td>VPD U vectors</td><td>0.4808</td></tr>
<tr><td>VPD V vectors</td><td>0.5156</td></tr>
<tr><td>PLT (e2e)</td><td>0.4390</td></tr>
<tr><td>CLT (e2e, parallel)</td><td>0.3468</td></tr>
<tr><td>TC (local MSE)</td><td>0.8063</td></tr>
<tr><td>CLT (local MSE)</td><td>0.6078</td></tr>
<tr><td>VPD rank-1 (V@U)</td><td>0.2826</td></tr>
<tr><td>VPD clusters (rank-N, cross-model)</td><td>0.3181</td></tr>
<tr><td>(Baseline) VPD at init U vectors</td><td>0.1263</td></tr>
<tr><td>(Baseline) VPD at init V vectors</td><td>0.1300</td></tr>
<tr><td>(Baseline) VPD at init rank-1 (V@U)</td><td>0.0122</td></tr>
</table>


Overall, we're uncertain how much emphasis to put on these similarities. While it is naturally appealing to think that there is a single 'correct' decomposition, we are not sure that this intuition fully accounts for the extent of the degeneracy in neural networks. One of the reasons that neural networks are so good at learning is the sheer amount of degeneracy they seem to have: It is easier to find a good solution in a space where there are many good solutions! It seems quite possible that, even though we place a number of constraints on the solution that VPD looks for, there is not just one set of ground truth mechanisms, but in fact an entire space of optimal parameter components that are nonetheless mechanistically faithful! The same is true of dictionary learning approaches. While (all else equal) cross seed consistency is a desirable property of a decomposition method, other properties such as mechanistic faithfulness are probably closer to what we want our methods to achieve.


### Stochastic vs. adversarial training loss

<label id="app:decomp-stats"/>


The adversarial loss greatly improves the decomposition performance for small source ($r$) values (<ref>fig:adv-vs-no-adv</ref>).

<figure>
<label id="fig:adv-vs-no-adv"/>
<img src="figures/adv_vs_no_adv.png">
<figcaption>Comparison between a decomposition with and without adversarial loss. The training configuration is otherwise identical. The CE loss is especially improved for small values of $r$.</figcaption>
</figure>

<!--DAN: The note that says "r=0->CI masks" is confusing. It can be read as "r goes from 0 on the left to CI masks on the right". I'd maybe just remove that note and put a note in the caption saying that r=0 is the same as using the CI masks. Or use r=0 (i.e. CI masks) and r=1 (i.e. unamsked) instead. -->



### OV circuit metric: Data-weighted Frobenius cosine similarity

<label id="app:OV-metric-data-frob"/>

<!-- TODO(Lee)(High priority) finish cleaning up appendix attention methods -->

To study the OV circuit across multiple heads, it is helpful think of $W_{OV}^h$ in terms of its singular value decomposition: $W_{OV}^h = \boldsymbol{L} \boldsymbol{S} \boldsymbol{R}^\top $. Now, we construct two new matrices for each $W_{OV}^h$ matrix:

$$
M^{\text{read}}_h = {W_{OV}^h}^\top W_{OV}^h = \boldsymbol{R} \boldsymbol{S}^2 \boldsymbol{R}^\top$$
$$M^{\text{write}}_h = W_{OV}^h {W_{OV}^h}^\top = \boldsymbol{L} \boldsymbol{S}^2 \boldsymbol{L}^\top$$

We can study how much each head reads and writes to the same subspace by comparing the similarity between the $M^{\text{read or write}}_h$ matrices of different heads. We compare them using a metric called the **Frobenius cosine similarity**, which is a cosine similarity metric for matrices: 
<!-- TODO(Lee)(High priority) write the missing math symbols in the line above -->

$$ S(M_a, M_b) = \frac{\langle M_a, M_b \rangle_F}{\|M_a\|_F \|M_b\|_F} $$

We will also measure the Frobenius cosine similarity between the raw $W_{OV}^h$ matrices of each head, since it is possible that even though matrices might read from and write to similar subspaces, their singular vectors might be paired differently. 

How should we understand this metric? On an intuitive level, we can think of a given $W_{OV}^h$ matrix's read- or write-subspace as a $d_{\text{head}}$-dimensional ellipsoid in $\mathbb{R}^{d_{\text{model}}}$ space, where the axes of the ellipsoid are the scaled right or left singular vectors of $W_{OV}^h$ matrix respectively. The Frobenius cosine similarity measures how much the read- or write-ellipsoid of one head overlaps with another head's. If the ellipsoids perfectly overlap, then the Frobenius cosine similarity is 1. If they exist in entirely non-overlapping subspaces, then their Frobenius cosine similarity is 0. For comparison purposes, we'll compare the Frobenius cosine similarities with a random matrix baseline. This will help us understand whether the model has learned to use more or less overlapping subspaces than would be expected for a pair of random matrices of the same size and dimension.

<!-- A random matrix of the same rank and dimensionality has an expected Frobenius cosine similarity of ~$0.143$ (<ref>app:expected_frob_proof</ref>).  -->


**Weighting subspaces by data variation**

However, the raw Frobenius cosine similarity between these matrices may potentially be misleading. The network does not use every subspace equally. Some subspaces might not contain much of the activations. Unless our metric accounts for how much of the activations lie within the subspaces that the $W_{OV}$ matrices read from and write to, we may get a misleading sense of how similar a pair of heads is. We should therefore weight different dimensions according to the amount of activation variation that exists along that axis. 

To do this, we form the **data-weighted** value matrix for each head. For a dataset of activations $X$, we perform PCA to get the principal axes of variation $\bar{Z}$:

$$\bar{X} = X - \bm{1} \bm{\mu}^\top, \qquad \bar{X} = \bar{U} \bar{S} \bar{Z}^\top,$$

$$\bar{U} \in \mathbb{R}^{n \times d_{\text{model}}}, \quad \bar{S} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}, \quad \bar{Z}^\top \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$
<!-- LaTeX original:
\bar{X} = X - \bm{1} \bm{\mu}^\top, \qquad \bar{X} = \bar{U} \bar{S} \bar{Z}^\top,
-->

where $\bm{\mu} = \frac{1}{N}\sum_n \bm{x}_n$. We then project $W_{OV}^h$ onto the data's principal axes of variation and scale each axis by the corresponding singular value, yielding the data-weighted value projection matrix for head $h$:


$$W_{OV}^{h, X} = W_{OV}^h \bar{Z} \bar{S}.$$

We can now construct data-weighted read and write Gram matrices ($M^{X, \text{read}}_h$ and $M^{X, \text{read}}_h$) as described above for the data-*un*weighted case. We can then use the Frobenius cosine similarity between them to understand how similarly the OV circuit of each head reads and write the actual data that it sees. 

We can also use this approach to *selectively* study how the OV circuit interacts with particular QK pairs. If we filter the dataset such that it contains only datapoints where the associated K subcomponent is causally important, then we can understand whether those pairs are moving similar or dissimilar value information in each head! 

We should note that the pair of subcomponents involved in previous token behavior are almost always active, and so we don't get to benefit from this QK-based filtering approach. But later we will study another behavior that is conditionally active, where it will be beneficial to understand how similar the OV circuits in each head behave only when that QK subcomponent interaction is active. 

We are now equipped enough to return to our analysis of previous token behavior and study its OV circuit to establish whether its heads attend to similar or distinct residual stream subspaces.


### Expected Frobenius Cosine Similarity of Random Low-Rank Gram Matrices {toc: Random baseline for Gram matrix cosine similarity}

<label id="app:expected_frob_proof"/>



In this section, we derive the expected Frobenius cosine similarity between the Gram matrices of two randomly initialized attention heads. We provide a proof and discuss an empirical test, which both agree.

##### Proof: Standard (unweighted) Frobenius cosine similarity


Let $W_a, W_b \in \mathbb{R}^{k \times d}$ be the value projection matrices for two attention heads, where $d$ is the model dimension ($d_{\text{model}}$) and $k$ is the head dimension ($d_{\text{head}}$). We initialize the elements of $W_a$ and $W_b$ independently from a standard normal distribution, $\mathcal{N}(0, 1)$. 

We define the corresponding Gram matrices as $M_a = W_a^\top W_a$ and $M_b = W_b^\top W_b$. Notice that while $M_a, M_b \in \mathbb{R}^{d \times d}$, their rank is bounded by $k$.

The Frobenius cosine similarity between $M_a$ and $M_b$ is defined as:
$$
S(M_a, M_b) = \frac{\langle M_a, M_b \rangle_F}{\|M_a\|_F \|M_b\|_F} = \frac{\operatorname{tr}(M_a M_b)}{\|M_a\|_F \|M_b\|_F}
$$

By the Law of Large Numbers in high-dimensional spaces, the variance of the matrix norms is small relative to their expectation. Therefore, we can approximate the expected value of the ratio by the ratio of the expected values:
$$
\mathbb{E}[S(M_a, M_b)] \approx \frac{\mathbb{E}[\operatorname{tr}(M_a M_b)]}{\mathbb{E}[\|M\|_F^2]}
$$
where we use the fact that $\mathbb{E}[\|M_a\|_F] \approx \sqrt{\mathbb{E}[\|M_a\|_F^2]}$. 

Because $W_a$ has elements drawn from $\mathcal{N}(0, 1)$, its Gram matrix $M_a$ follows a standard Wishart distribution with $k$ degrees of freedom, denoted as $M_a \sim \mathcal{W}_d(k, I_d)$.

**The Expected Inner Product (Numerator)**

The expected value of a Wishart matrix $\mathcal{W}_d(k, I_d)$ is $k I_d$. Because $W_a$ and $W_b$ are independent, $M_a$ and $M_b$ are also independent. Therefore, the expectation of their inner product is the trace of the product of their expectations:

$$\begin{aligned}
\mathbb{E}[\operatorname{tr}(M_a M_b)] &= \operatorname{tr}(\mathbb{E}[M_a]\mathbb{E}[M_b]) \\
&= \operatorname{tr}((k I_d)(k I_d)) \\
&= k^2 \operatorname{tr}(I_d) \\
&= k^2 d
\end{aligned}
$$

**The Expected Squared Norm (Denominator)**

To find $\mathbb{E}[\|M\|_F^2]$, we sum the expected squared values of all elements in the Gram matrix. Let $M_{ij}$ be the entry in the $i$-th row and $j$-th column.
$$
\|M\|_F^2 = \sum_{i=1}^d \sum_{j=1}^d M_{ij}^2 = \sum_{i=1}^d M_{ii}^2 + \sum_{i \neq j} M_{ij}^2
$$

**Diagonal Elements**

The diagonal elements are $M_{ii} = \sum_{r=1}^k W_{ri}^2$. Since $W_{ri} \sim \mathcal{N}(0, 1)$, each $M_{ii}$ follows a Chi-squared distribution with $k$ degrees of freedom ($\chi^2_k$).
The mean of a $\chi^2_k$ variable is $k$, and its variance is $2k$. Using the identity $\mathbb{E}[X^2] = \operatorname{Var}(X) + (\mathbb{E}[X])^2$, we have:
$$
\mathbb{E}[M_{ii}^2] = 2k + k^2
$$
Since there are $d$ diagonal elements, their total contribution is $d(k^2 + 2k)$.

**Off-Diagonal Elements**

The off-diagonal elements are $M_{ij} = \sum_{r=1}^k W_{ri} W_{rj}$ for $i \neq j$. Because $W_{ri}$ and $W_{rj}$ are independent standard normal variables, their product has a mean of $0$ and a variance of $1$. The sum of $k$ such independent terms has a mean of $0$ and a variance of $k$. Thus:
$$
\mathbb{E}[M_{ij}^2] = \operatorname{Var}(M_{ij}) + (\mathbb{E}[M_{ij}])^2 = k + 0 = k
$$
There are $d(d-1)$ off-diagonal elements, so their total contribution is $d(d-1)k$.

**Total Expected Squared Norm**

Combining the diagonal and off-diagonal contributions yields:
$$\begin{aligned}
\mathbb{E}[\|M\|_F^2] &= d(k^2 + 2k) + d(d-1)k \\
&= dk^2 + 2dk + d^2k - dk \\
&= dk(d + k + 1)
\end{aligned}$$

**Final Expected Baseline**

Substituting the expected inner product and the expected squared norm back into our similarity approximation:
$$
\mathbb{E}[S(M_a, M_b)] \approx \frac{k^2 d}{dk(d + k + 1)} = \frac{k}{d + k + 1}
$$

For our specific architecture, the model dimension is $d = 768$ and the head dimension is $k = 128$. Plugging these values into the derived formula gives the exact expected random baseline for the subspace overlap:
$$
\mathbb{E}[S(M_a, M_b)] \approx \frac{128}{768 + 128 + 1} = \frac{128}{897} \approx 0.1427
$$

Thus, we have $\approx 0.1427$ as the expected baseline for the Frobenius cosine similarity between two randomly initialized heads of this dimension and rank.

This value exactly matches an empirical random baseline computed via Monte Carlo simulation:

##### Empirical: Standard (unweighted) Frobenius cosine similarity

We generate 1000 pairs of random matrices $W_a, W_b \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}$ with i.i.d. standard normal entries, compute their Gram matrices $M = W^\top W$, and calculate the Frobenius cosine similarity $\frac{\text{tr}(M_a M_b)}{\lVert M_a \rVert_F \lVert M_b \rVert_F}$ for each pair. The mean across pairs gives the expected overlap between matrices with no structural relationship. The empirical result exactly matched the theoretical result proved above (0.1427). 

##### Data-weighted Frobenius cosine similarity

For the data-weighted case, use the same Monte Carlo procedure, but right-multiply each random matrix by $\bar{Z}\bar{S}$ (the right singular vectors scaled by singular values from the mean-centered data) before computing Gram matrices. This ensures the baseline reflects the anisotropy of the residual stream because, in a low-rank data distribution, even unrelated matrices may exhibit elevated subspace overlap. This resulted in a higher baseline: 0.564.

### Layer 1 K and V subcomponent relations

<figure>
<label id="fig:pkv"/>
<img src="figures/pkv_layer1.png">
<figcaption>The probability of each K subcomponent being active when a given V component is active. This tells us what K components are primarily responsible for moving particular kinds of values. The <comp key>1.attn.k:329</comp> subcomponent is always active, and therefore moves all kinds of value components. </figcaption>
</figure>

<figure class="wide">
<label id="fig:prev_tok_ov_overlap_k_119"/>
<img src="figures/layer1_ov_paper_figure_k_119.png">
<figcaption>Data-weighted cosine similarities between each head's $W_{OV}^h$ read- and write matrices, and the cosine similarity between each heads raw $W_{OV}^h$. Here, data-weighting uses data where subcomponent <comp key>1.attn.k:119</comp> is causally important. </figcaption>
</figure>


### Layer 1 O, V subcomponents most aligned with attention heads on data where Layer 1 K.119 is causally important {toc: Layer 1 subcomponents most aligned with heads}

<label id="app:ov-alignment-k119"/>

Here we list, for each attention head, the top-20 V components and top-20 O components whose subcomponents are most aligned with that head's
OV circuit.

Here, alignment $= ||W v_{c}^{\text{scaled}}||$ (read) or $||W^\top u_{c}^{\text{scaled}}||$ (write), where vectors are scaled by the norm of the other factor in the rank-1 decomposition.


Top 20 V components (read-aligned) and O components (write-aligned) per head.

##### Head 0

**Read-aligned V components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | v.22 | 74.9453 | punctuation, syntax, and formatting tokens |
| 2 | v.984 | 68.3169 | fires on punctuation and symbols |
| 3 | v.1000 | 64.4708 | fires on punctuation, delimiters, and structural boundaries |
| 4 | v.346 | 62.6574 | distinguishes function words (positive) and content words (negative) |
| 5 | v.568 | 59.1917 | fires on word prefixes and partial words |
| 6 | v.745 | 56.6575 | formatting symbols, operators, and spatial alignment |
| 7 | v.946 | 54.1788 | distinguishes content words from function words/symbols |
| 8 | v.452 | 52.9072 | predicts syntax and punctuation after code identifiers/attributes |
| 9 | v.428 | 51.1282 | fragments of proper nouns, foreign, and technical words |
| 10 | v.531 | 50.0481 | opening parentheses, brackets, braces, and quotes |
| 11 | v.315 | 48.1878 | adjectives and punctuation separators |
| 12 | v.340 | 47.6278 | syntactic linkages and prepositions |
| 13 | v.725 | 45.9728 | fires on numbers and digits |
| 14 | v.72 | 45.8301 | fires on punctuation to predict newlines and connectors |
| 15 | v.940 | 45.5767 | fires on line breaks and sequence boundaries |
| 16 | v.88 | 44.7435 | scientific, medical, technical, and academic terminology |
| 17 | v.494 | 44.1038 | predicts line breaks or indentation in formatted text |
| 18 | v.694 | 43.9273 | fires on pronouns related to people |
| 19 | v.228 | 42.9872 | fires broadly on nouns, verbs, and adjectives in text |
| 20 | v.195 | 41.3329 | fires on delimiters and structural punctuation |

**Write-aligned O components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | o.923 | 231.5484 | first token of the sequence |
| 2 | o.411 | 203.4490 | code, markup, and technical formatting syntax |
| 3 | o.630 | 191.1107 | punctuation, symbols, and syntax in technical text |
| 4 | o.753 | 181.5993 | closing parentheses and brackets in code and math |
| 5 | o.300 | 173.9110 | code and structured text syntax/indentation |
| 6 | o.180 | 167.1310 | diffuse firing on tokens within words/phrases |
| 7 | o.311 | 162.3423 | fires universally on most tokens |
| 8 | o.362 | 160.9102 | fires on names, citations, proper nouns and formatting tokens |
| 9 | o.37 | 144.1937 | continuations of multi-token entities and compound words |
| 10 | o.860 | 140.8711 | structural and formatting markers vs content words |
| 11 | o.578 | 135.6885 | heterogeneous component / lack of clear pattern |
| 12 | o.292 | 118.3135 | fires broadly on various tokens, promoting line breaks and punctuation |
| 13 | o.91 | 116.1987 | elements and separators in lists |
| 14 | o.113 | 111.6550 | punctuation, brackets, and mathematical symbols |
| 15 | o.886 | 111.2655 | identifiers in citations, references, and urls |
| 16 | o.986 | 105.6721 | sentence/paragraph boundaries and transition words |
| 17 | o.336 | 103.8562 | diverges between function words and complex technical terms |
| 18 | o.117 | 101.4838 | predicts 'of' (and other function words) after nouns |
| 19 | o.866 | 98.8138 | predicts newlines and separators at line ends |
| 20 | o.707 | 94.0369 | delimiters and subword boundaries in structured text |

##### Head 1

**Read-aligned V components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | v.984 | 59.9101 | fires on punctuation and symbols |
| 2 | v.346 | 56.6177 | distinguishes function words (positive) and content words (negative) |
| 3 | v.22 | 52.5358 | punctuation, syntax, and formatting tokens |
| 4 | v.1000 | 52.4579 | fires on punctuation, delimiters, and structural boundaries |
| 5 | v.946 | 45.8153 | distinguishes content words from function words/symbols |
| 6 | v.531 | 44.4541 | opening parentheses, brackets, braces, and quotes |
| 7 | v.745 | 42.9024 | formatting symbols, operators, and spatial alignment |
| 8 | v.428 | 42.6691 | fragments of proper nouns, foreign, and technical words |
| 9 | v.452 | 41.8698 | predicts syntax and punctuation after code identifiers/attributes |
| 10 | v.315 | 41.2963 | adjectives and punctuation separators |
| 11 | v.340 | 41.1914 | syntactic linkages and prepositions |
| 12 | v.568 | 40.6832 | fires on word prefixes and partial words |
| 13 | v.940 | 40.6337 | fires on line breaks and sequence boundaries |
| 14 | v.494 | 40.3412 | predicts line breaks or indentation in formatted text |
| 15 | v.550 | 37.5125 | underscores in code, c++ accessors, and math sub/superscripts |
| 16 | v.115 | 37.2987 | opening quotes, asterisks, and formatting marks |
| 17 | v.1014 | 36.1067 | subordinating conjunctions and relative pronouns |
| 18 | v.910 | 36.1015 | single letters and math variables |
| 19 | v.228 | 35.7553 | fires broadly on nouns, verbs, and adjectives in text |
| 20 | v.72 | 35.2095 | fires on punctuation to predict newlines and connectors |

**Write-aligned O components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | o.362 | 167.0992 | fires on names, citations, proper nouns and formatting tokens |
| 2 | o.490 | 164.5913 | line start and indentation tokens |
| 3 | o.337 | 149.7576 | fires inside parentheses or mathematical formulas |
| 4 | o.895 | 132.7422 | variables, math symbols, and syntax in technical text |
| 5 | o.986 | 116.9061 | sentence/paragraph boundaries and transition words |
| 6 | o.311 | 116.6149 | fires universally on most tokens |
| 7 | o.338 | 113.4746 | word continuations and compound word fragments |
| 8 | o.860 | 110.9601 | structural and formatting markers vs content words |
| 9 | o.630 | 110.7552 | punctuation, symbols, and syntax in technical text |
| 10 | o.37 | 106.5099 | continuations of multi-token entities and compound words |
| 11 | o.807 | 104.1172 | fires on fragments of identifiers and numbers |
| 12 | o.117 | 103.3490 | predicts 'of' (and other function words) after nouns |
| 13 | o.31 | 103.1152 | punctuation and math symbols in code and math |
| 14 | o.907 | 98.6125 | predicts continuations of collocations and common phrases |
| 15 | o.77 | 98.0239 | punctuation, numbers, and subwords in technical text |
| 16 | o.573 | 96.1810 | general syntax and varied text prediction |
| 17 | o.285 | 96.0340 | activates on newlines and indentation |
| 18 | o.344 | 90.3763 | fires on components of hyphenated or compound words |
| 19 | o.928 | 89.7079 | general syntax token or end-of-phrase pattern processor |
| 20 | o.352 | 86.7081 | subword and identifier continuation |

##### Head 2

**Read-aligned V components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | v.984 | 83.9813 | fires on punctuation and symbols |
| 2 | v.22 | 79.5923 | punctuation, syntax, and formatting tokens |
| 3 | v.346 | 69.2600 | distinguishes function words (positive) and content words (negative) |
| 4 | v.1000 | 68.2968 | fires on punctuation, delimiters, and structural boundaries |
| 5 | v.531 | 66.0043 | opening parentheses, brackets, braces, and quotes |
| 6 | v.725 | 59.3996 | fires on numbers and digits |
| 7 | v.946 | 58.8498 | distinguishes content words from function words/symbols |
| 8 | v.428 | 58.5928 | fragments of proper nouns, foreign, and technical words |
| 9 | v.568 | 57.8588 | fires on word prefixes and partial words |
| 10 | v.452 | 56.9823 | predicts syntax and punctuation after code identifiers/attributes |
| 11 | v.340 | 54.3899 | syntactic linkages and prepositions |
| 12 | v.315 | 51.5444 | adjectives and punctuation separators |
| 13 | v.745 | 51.5016 | formatting symbols, operators, and spatial alignment |
| 14 | v.940 | 50.3622 | fires on line breaks and sequence boundaries |
| 15 | v.494 | 49.3918 | predicts line breaks or indentation in formatted text |
| 16 | v.389 | 47.9647 | delimiters and punctuation in structured text and code |
| 17 | v.88 | 46.4945 | scientific, medical, technical, and academic terminology |
| 18 | v.910 | 46.3691 | single letters and math variables |
| 19 | v.188 | 46.0352 | structural punctuation and syntax symbols |
| 20 | v.919 | 45.8033 | fires on newlines and indentation |

**Write-aligned O components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | o.923 | 432.9877 | first token of the sequence |
| 2 | o.578 | 249.4896 | heterogeneous component / lack of clear pattern |
| 3 | o.180 | 227.2081 | diffuse firing on tokens within words/phrases |
| 4 | o.866 | 208.1991 | predicts newlines and separators at line ends |
| 5 | o.336 | 205.5378 | diverges between function words and complex technical terms |
| 6 | o.707 | 188.7079 | delimiters and subword boundaries in structured text |
| 7 | o.292 | 167.8825 | fires broadly on various tokens, promoting line breaks and punctuation |
| 8 | o.311 | 154.5922 | fires universally on most tokens |
| 9 | o.117 | 150.1182 | predicts 'of' (and other function words) after nouns |
| 10 | o.37 | 140.2300 | continuations of multi-token entities and compound words |
| 11 | o.753 | 139.6652 | closing parentheses and brackets in code and math |
| 12 | o.113 | 139.5260 | punctuation, brackets, and mathematical symbols |
| 13 | o.362 | 137.9804 | fires on names, citations, proper nouns and formatting tokens |
| 14 | o.630 | 137.9800 | punctuation, symbols, and syntax in technical text |
| 15 | o.860 | 130.5145 | structural and formatting markers vs content words |
| 16 | o.300 | 129.2226 | code and structured text syntax/indentation |
| 17 | o.867 | 119.7837 | miscellaneous text and punctuation predictor |
| 18 | o.319 | 110.2475 | numbers, units, math, and structured quantitative data |
| 19 | o.580 | 109.0233 | fires on the first token of sequences |
| 20 | o.202 | 102.8278 | subwords in camelcase and capitalized words |

##### Head 3

**Read-aligned V components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | v.984 | 70.1569 | fires on punctuation and symbols |
| 2 | v.745 | 57.7404 | formatting symbols, operators, and spatial alignment |
| 3 | v.946 | 53.8508 | distinguishes content words from function words/symbols |
| 4 | v.494 | 53.3743 | predicts line breaks or indentation in formatted text |
| 5 | v.346 | 52.9415 | distinguishes function words (positive) and content words (negative) |
| 6 | v.1000 | 51.4397 | fires on punctuation, delimiters, and structural boundaries |
| 7 | v.22 | 49.8140 | punctuation, syntax, and formatting tokens |
| 8 | v.428 | 49.2318 | fragments of proper nouns, foreign, and technical words |
| 9 | v.940 | 47.6670 | fires on line breaks and sequence boundaries |
| 10 | v.731 | 43.9993 | fires on forms of 'to be' |
| 11 | v.568 | 42.7718 | fires on word prefixes and partial words |
| 12 | v.315 | 41.9959 | adjectives and punctuation separators |
| 13 | v.228 | 39.7090 | fires broadly on nouns, verbs, and adjectives in text |
| 14 | v.340 | 39.2528 | syntactic linkages and prepositions |
| 15 | v.725 | 39.1140 | fires on numbers and digits |
| 16 | v.72 | 38.4603 | fires on punctuation to predict newlines and connectors |
| 17 | v.195 | 38.0045 | fires on delimiters and structural punctuation |
| 18 | v.629 | 37.8436 | predicts word completions from initial letters |
| 19 | v.452 | 37.7656 | predicts syntax and punctuation after code identifiers/attributes |
| 20 | v.389 | 37.0761 | delimiters and punctuation in structured text and code |

**Write-aligned O components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | o.311 | 333.3696 | fires universally on most tokens |
| 2 | o.630 | 263.6867 | punctuation, symbols, and syntax in technical text |
| 3 | o.37 | 257.2931 | continuations of multi-token entities and compound words |
| 4 | o.300 | 251.2589 | code and structured text syntax/indentation |
| 5 | o.180 | 177.4386 | diffuse firing on tokens within words/phrases |
| 6 | o.292 | 166.7455 | fires broadly on various tokens, promoting line breaks and punctuation |
| 7 | o.362 | 147.2889 | fires on names, citations, proper nouns and formatting tokens |
| 8 | o.552 | 111.1569 | fires on the first token of a sequence |
| 9 | o.860 | 111.0820 | structural and formatting markers vs content words |
| 10 | o.986 | 109.8089 | sentence/paragraph boundaries and transition words |
| 11 | o.91 | 105.0967 | elements and separators in lists |
| 12 | o.411 | 101.4439 | code, markup, and technical formatting syntax |
| 13 | o.113 | 98.3131 | punctuation, brackets, and mathematical symbols |
| 14 | o.266 | 85.9204 | fires on latex macros and mathematical symbols |
| 15 | o.340 | 85.5521 | punctuation, math, code, formatting symbols, and short affixes |
| 16 | o.285 | 81.8968 | activates on newlines and indentation |
| 17 | o.923 | 77.5562 | first token of the sequence |
| 18 | o.117 | 75.9101 | predicts 'of' (and other function words) after nouns |
| 19 | o.886 | 74.8477 | identifiers in citations, references, and urls |
| 20 | o.480 | 74.4309 | urls, file paths, and namespaces |

##### Head 4

**Read-aligned V components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | v.346 | 146.1802 | distinguishes function words (positive) and content words (negative) |
| 2 | v.22 | 145.5668 | punctuation, syntax, and formatting tokens |
| 3 | v.984 | 142.4020 | fires on punctuation and symbols |
| 4 | v.745 | 129.9418 | formatting symbols, operators, and spatial alignment |
| 5 | v.1000 | 116.1570 | fires on punctuation, delimiters, and structural boundaries |
| 6 | v.568 | 113.2827 | fires on word prefixes and partial words |
| 7 | v.452 | 111.5915 | predicts syntax and punctuation after code identifiers/attributes |
| 8 | v.910 | 110.8966 | single letters and math variables |
| 9 | v.946 | 110.8379 | distinguishes content words from function words/symbols |
| 10 | v.428 | 110.4635 | fragments of proper nouns, foreign, and technical words |
| 11 | v.257 | 109.5629 | letters a, b, c, d and their continuations |
| 12 | v.88 | 108.0470 | scientific, medical, technical, and academic terminology |
| 13 | v.69 | 106.0124 | fires on whitespace and indentation |
| 14 | v.340 | 105.9949 | syntactic linkages and prepositions |
| 15 | v.195 | 105.3308 | fires on delimiters and structural punctuation |
| 16 | v.389 | 101.5132 | delimiters and punctuation in structured text and code |
| 17 | v.136 | 99.8001 | verbs |
| 18 | v.1014 | 99.7338 | subordinating conjunctions and relative pronouns |
| 19 | v.72 | 99.5376 | fires on punctuation to predict newlines and connectors |
| 20 | v.190 | 99.4277 | variable declarations, latex math, and markdown markers |

**Write-aligned O components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | o.753 | 832.6620 | closing parentheses and brackets in code and math |
| 2 | o.630 | 591.0163 | punctuation, symbols, and syntax in technical text |
| 3 | o.411 | 538.0296 | code, markup, and technical formatting syntax |
| 4 | o.860 | 522.9536 | structural and formatting markers vs content words |
| 5 | o.292 | 503.4616 | fires broadly on various tokens, promoting line breaks and punctuation |
| 6 | o.923 | 462.0907 | first token of the sequence |
| 7 | o.180 | 434.1415 | diffuse firing on tokens within words/phrases |
| 8 | o.300 | 407.3344 | code and structured text syntax/indentation |
| 9 | o.311 | 342.3619 | fires universally on most tokens |
| 10 | o.340 | 295.8672 | punctuation, math, code, formatting symbols, and short affixes |
| 11 | o.578 | 274.5031 | heterogeneous component / lack of clear pattern |
| 12 | o.113 | 268.6180 | punctuation, brackets, and mathematical symbols |
| 13 | o.473 | 255.1414 | promotes common stop words in continuous prose |
| 14 | o.37 | 227.6168 | continuations of multi-token entities and compound words |
| 15 | o.336 | 212.8854 | diverges between function words and complex technical terms |
| 16 | o.756 | 211.6430 | fires in various technical texts |
| 17 | o.362 | 206.5355 | fires on names, citations, proper nouns and formatting tokens |
| 18 | o.886 | 171.3670 | identifiers in citations, references, and urls |
| 19 | o.866 | 167.9524 | predicts newlines and separators at line ends |
| 20 | o.117 | 165.4990 | predicts 'of' (and other function words) after nouns |

##### Head 5

**Read-aligned V components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | v.1000 | 57.2500 | fires on punctuation, delimiters, and structural boundaries |
| 2 | v.946 | 56.3107 | distinguishes content words from function words/symbols |
| 3 | v.346 | 53.8432 | distinguishes function words (positive) and content words (negative) |
| 4 | v.428 | 51.4916 | fragments of proper nouns, foreign, and technical words |
| 5 | v.984 | 50.2268 | fires on punctuation and symbols |
| 6 | v.22 | 50.2116 | punctuation, syntax, and formatting tokens |
| 7 | v.88 | 45.2513 | scientific, medical, technical, and academic terminology |
| 8 | v.340 | 43.3620 | syntactic linkages and prepositions |
| 9 | v.568 | 42.4811 | fires on word prefixes and partial words |
| 10 | v.72 | 42.2889 | fires on punctuation to predict newlines and connectors |
| 11 | v.315 | 42.0921 | adjectives and punctuation separators |
| 12 | v.228 | 41.7201 | fires broadly on nouns, verbs, and adjectives in text |
| 13 | v.724 | 40.7481 | capitalized and uppercase words/tokens |
| 14 | v.725 | 40.1933 | fires on numbers and digits |
| 15 | v.531 | 39.5459 | opening parentheses, brackets, braces, and quotes |
| 16 | v.745 | 39.3306 | formatting symbols, operators, and spatial alignment |
| 17 | v.494 | 39.0761 | predicts line breaks or indentation in formatted text |
| 18 | v.940 | 37.3873 | fires on line breaks and sequence boundaries |
| 19 | v.195 | 36.8701 | fires on delimiters and structural punctuation |
| 20 | v.232 | 36.8621 | fires on tokens ending in 's', especially plurals |

**Write-aligned O components (top 20)**

| Rank | Comp | Alignment | Label |
|------|------|-----------|-------|
| 1 | o.923 | 227.1903 | first token of the sequence |
| 2 | o.311 | 178.6259 | fires universally on most tokens |
| 3 | o.630 | 160.5296 | punctuation, symbols, and syntax in technical text |
| 4 | o.180 | 156.3924 | diffuse firing on tokens within words/phrases |
| 5 | o.300 | 144.0436 | code and structured text syntax/indentation |
| 6 | o.37 | 138.1541 | continuations of multi-token entities and compound words |
| 7 | o.578 | 134.7332 | heterogeneous component / lack of clear pattern |
| 8 | o.292 | 125.1081 | fires broadly on various tokens, promoting line breaks and punctuation |
| 9 | o.860 | 122.2877 | structural and formatting markers vs content words |
| 10 | o.362 | 114.0454 | fires on names, citations, proper nouns and formatting tokens |
| 11 | o.336 | 113.8206 | diverges between function words and complex technical terms |
| 12 | o.117 | 79.2022 | predicts 'of' (and other function words) after nouns |
| 13 | o.411 | 77.4504 | code, markup, and technical formatting syntax |
| 14 | o.340 | 75.1049 | punctuation, math, code, formatting symbols, and short affixes |
| 15 | o.886 | 71.0343 | identifiers in citations, references, and urls |
| 16 | o.986 | 70.3552 | sentence/paragraph boundaries and transition words |
| 17 | o.113 | 70.1237 | punctuation, brackets, and mathematical symbols |
| 18 | o.707 | 68.6813 | delimiters and subword boundaries in structured text |
| 19 | o.814 | 68.2534 | fires on the second item in a coordinated pair |
| 20 | o.163 | 66.0070 | legal document citations and abbreviations |

### Interaction graphs


#### Gradient attributions

<label id="app:gradient_attributions"/>

To understand how subcomponents interact with each other during the forward pass, we compute gradient attributions between pairs of subcomponents at adjacent layers in the computational graph. These attributions form the edges of an interaction graph that visualizes the flow of information through the decomposed model on a given prompt or aggregated over the dataset.

Recall that each subcomponent $c$ at layer $l$ has a *subcomponent activation* $a^l_{b,t,c} = (\vec{V^l_c})^\top \vec{h^l_{b,t}}$, where $\vec{h^l_{b,t}}$ is the pre-weight activation vector at layer $l$ on batch element $b$ at sequence position $t$. This is the projection of the input onto the right singular vector of the rank-one subcomponent, and it determines how strongly the subcomponent contributes to the layer's output.

For a source subcomponent $c_1$ at layer $l_1$ and a target subcomponent $c_2$ at layer $l_2$ (where $l_1$ feeds into $l_2$ in the computational graph), we define the *gradient attribution* on batch element $b$ and at source sequence position $t_1$ and target sequence position $t_2$ as:


$$\alpha(c_1 \to c_2; b, t_1, t_2) = \frac{\partial a^{l_2}_{b, t_2, c_2}}{\partial a^{l_1}_{b,t_1, c_1}} \cdot a^{l_1}_{b,t_1, c_1} \cdot g^{l_1}_{b,t_1,c_1}$$
<!-- LaTeX original:
\alpha(c_s \to c_t; x, t_s, t_t) = \frac{\partial a^{l_t}_{c_t}(x,t_t)}{\partial
 a^{l_s}_{c_s}(x,t_s)} \cdot a^{l_s}_{c_s}(x,t_s) \cdot g^{l_s}_{c_s}(x,t_s)
-->

where $g^{l_1}_{b,t_1,c_1}$ is the causal importance of the source subcomponent. The gradient $\times$ activation product $ \frac{\partial a^{l_2}_{b,t_2, c_2}}{\partial a^{l_1}_{b,t_1, c_1}} \cdot a^{l_1}_{b,t_1, c_1}$ gives a first-order estimate of how much the source subcomponent's activation contributes to the target subcomponent's activation. Weighting by the causal importance $g^{l_1}_{b,t_1,c_1}$ ensures that subcomponents which are not causally important on a given datapoint (i.e., those that can be ablated without affecting the output) do not contribute to the attribution, even if they happen to have nonzero activations and gradients.

For most adjacent layer pairs, source and target positions coincide ($t_1 = t_2$), since MLP and attention projection layers operate position-wise. However, for edges from key or value subcomponents to attention output subcomponents within the same attention layer, the source position $t_1$ can be any position up to and including the target position $t_2$ (i.e., $t_1 \leq t_2$, respecting the causal attention mask). This reflects the fact that key and value activations at earlier positions influence the attention output at later positions.



**Dataset-aggregated attributions.**

To obtain a summary of how subcomponents interact across the dataset, we aggregate attributions over all datapoints and all valid position pairs:


$$A(c_1 \to c_2) = \sum^B_{b=1} \sum_{t_1, t_2} \frac{\partial
 a^{l_2}_{b,t_2,c_2}}{\partial a^{l_1}_{b,t_1,c_1}} \cdot a^{l_1}_{b,t_1,c_1} \cdot
 g^{l_1}_{b,t_1,c_1}$$
<!-- LaTeX original:
A(c_s \to c_t) = \sum_{x \in \mathcal{D}} \sum_{t_s, t_t} \frac{\partial
 a^{l_t}_{c_t}(x,t_t)}{\partial a^{l_s}_{c_s}(x,t_s)} \cdot a^{l_s}_{c_s}(x,t_s) \cdot
 g^{l_s}_{c_s}(x,t_s)
-->
where the sum over $(t_1, t_2)$ ranges over $t_1 = t_2$ for position-wise layers and $t_1 \leq t_2$ for key/value-to-output edges in attention. In practice, we compute this sum over the training dataset using a distributed pipeline across multiple GPUs. To make attributions comparable across subcomponents with different activation scales and different frequencies of causal importance, we normalize by the total causal importance of the source and the root-mean-square activation of the target:


$$\hat{A}(c_1 \to c_2) = \frac{A(c_1 \to c_2)}{\left(\sum_{b,t_1} g^{l_1}_{b,t_1,c_1}\right) \cdot
 \text{RMS}(a^{l_2}_{c_2})}$$
<!-- LaTeX original:
\hat{A}(c_s \to c_t) = \frac{A(c_s \to c_t)}{\left(\sum_{x,t} g^{l_s}_{c_s}(x,t)\right) \cdot
\text{RMS}(a^{l_t}_{c_t})}
-->

where $\text{RMS}(a^{l_2}_{c_2}) = \sqrt{\frac{1}{BT} \sum_{b,t} (a^{l_2}_{b,t,c_2})^2}$ and $BT$ is the total number of tokens processed. Dividing by the source's cumulative causal importance puts the attribution on a per-occurrence scale (analogous to averaging over only the datapoints where the source is active), while dividing by the target's RMS activation accounts for the target's overall magnitude. Together, these normalizations allow meaningful comparison of attribution strengths across edges in the graph.

We also compute an absolute-value variant $A_{\text{abs}}(c_1 \to c_2)$, which replaces the target activation $a^{l_2}_{c_2}$ with its absolute value $|a^{l_2}_{c_2}|$ in the backward pass. This variant captures the total magnitude of influence irrespective of sign, and is useful for identifying strong interactions where the signed attribution may cancel across datapoints.



**Prompt-level attributions.**

For analyzing individual prompts, we compute position-aware attributions without aggregation. Given a prompt and a set of "alive" subcomponents (those with nonzero causal importance at each position), we compute the gradient attribution for each pair of alive source and target subcomponents at each valid combination of source and target positions. The resulting position-aware graph enables detailed analysis of how the model processes a specific input.

The main changes are: separate $t_s$ and $t_t$ indices throughout, an explicit paragraph explaining when they differ (K/V to O edges within an attention layer), and the dataset sum now ranges over valid position pairs rather than a single shared position.

#### Pruning for specific behaviors: Post-hoc causal importance optimization 

<label id="app:posthoc_ci"/>

During VPD base training, the causal importance function $\Gamma$ is trained to predict which subcomponents are necessary to reconstruct the target model's *full output distribution across all sequence positions*. However, when analyzing a specific behavior—such as the model's prediction of a particular token at a particular position—many causally important subcomponents will be irrelevant to that specific behavior, even though they are necessary for reconstructing the full output. To isolate only the subcomponents involved in a behavior of interest, we optimize new causal importance values *post hoc* on a single prompt, using a reconstruction loss that targets only the specific aspect of the output we wish to study.

##### Setup

Given a trained VPD model and a prompt, we first run the model's trained causal importance function to obtain the base causal importance values $g^l_{t,c}$ for all subcomponents on that prompt. We then identify the set of *alive* subcomponents for the prompt at each sequence position $t$: those for which $g^l_{t,c} > 0$. Only alive subcomponent causal importances are eligible for inclusion in the post-hoc optimization, though masks for the other subcomponents (and the $\Delta$ components) can still be sampled stochastically to ensure they remain ablatable. 

We parameterize the new causal importances using pre-sigmoid parameters $\phi^l_{t,c}$, one per alive subcomponent at each position. The causal importance values are obtained by passing these parameters through the same lower-leaky (for sampling the masks used in the forward pass) and upper leaky (for the $\mathcal{L}_{\text{importance-minimality}}$ and $\mathcal{L}_{\text{frequency-minimality}}$) hard sigmoid functions $\sigma_{H,\text{lower}}$, $\sigma_{H,\text{upper}}$ (see <ref>sec:vpd_ci_function</ref>) used during base training. The parameters $\phi^l_{t,c}$ are initialized to the pre-sigmoid values produced by the base causal importance function on this prompt, providing a warm start. Non-alive subcomponents have their causal importance fixed at zero throughout optimization.


##### Loss function

The post-hoc optimization minimizes a combination of a reconstruction loss and an importance minimality loss:

$$\mathcal{L}_{\text{post-hoc}} =\lambda_{\text{recon}} \cdot\mathcal{L}_{\text{recon}} + \lambda_{\text{min}} \cdot\mathcal{L}_{\text{importance-minimality}}$$
<!-- LaTeX original:
\mathcal{L}_{\text{post-hoc}} =\lambda_{\text{recon}} \cdot\mathcal{L}_{\text{recon}} + \lambda_{\text{min}} \cdot\mathcal{L}_{\text{importance-minimality}}
-->

The reconstruction loss $\mathcal{L}_{\text{recon}}$ is chosen to target the specific behavior of interest. For example, to study how the model predicts token $y$ at position $t^*$, we use a cross-entropy loss at that position:

$$\mathcal{L}_{\text{recon}} = -\log p_{\text{masked}}(y \mid x, t^*)$$
<!-- LaTeX original:
\mathcal{L}_{\text{recon}} = -\log p_{\text{masked}}(y \mid x, t^*)
-->

where $p_{\text{masked}}$ denotes the output distribution of the model with masks applied according to the post-hoc causal importances. Alternatively, if we wish to reconstruct the model's full output distribution at a specific position rather than targeting a particular token, we can use a KL-divergence loss:

$$\mathcal{L}_{\text{recon}} = D_{\text{KL}}\!\left(p_{\text{target}}(\cdot \mid x, t^*) \;\|\;
p_{\text{masked}}(\cdot \mid x, t^*)\right)$$
<!-- LaTeX original:
\mathcal{L}_{\text{recon}} = D_{\text{KL}}\!\left(p_{\text{target}}(\cdot \mid x, t^*) \;\|\;
p_{\text{masked}}(\cdot \mid x, t^*)\right)
-->

The importance minimality loss $\mathcal{L}_{\text{importance-minimality}}$ has the same form as in base training (<ref>eq:minimal</ref> and <ref>eq:freq_minimality</ref>), applied to the post-hoc causal importances $\tilde{g}^l_c(x,t)$. This loss encourages the optimization to find the sparsest set of subcomponents that can still reconstruct the targeted behavior. The coefficient $\lambda_{\text{min}}$ controls the sparsity–fidelity trade-off: larger values yield sparser graphs with fewer active subcomponents, potentially at the cost of reconstruction quality.

##### Masking during optimization


As in base training, the post-hoc causal importances define masks on the subcomponents via:

$$m^l_{t,c}(r) = \tilde{g}^l_{t,c} + (1 - \tilde{g}^l_{t,c}) r^l_{t,c}$$
<!-- LaTeX original:
m^l_c(x,t,r) = \tilde{g}^l_c(x,t) + (1 - \tilde{g}^l_c(x,t)) r^l_c(x,t)
-->


where $r^l_{t,c} \in [0,1]$. On each optimization step, we sample masks by drawing $r^l_{t,c}$, either stochastically uniformly or adversarially, and compute the reconstruction loss under those masks. This ensures that the optimization satisfies the same mechanistic faithfulness criterion as base training: subcomponents marked as unimportant must be ablatable in any combination without affecting the targeted output.

$$\mathcal{L}_{\text{post-hoc}} = \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{recon}}
+\lambda_{\text{min}} \cdot \mathcal{L}_{\text{importance-minimality}}
+ \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{adversarial-recon}}$$
<!-- LaTeX original:
\mathcal{L}_{\text{post-hoc}} = \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{recon}}
+\lambda_{\text{min}} \cdot \mathcal{L}_{\text{importance-minimality}}
+ \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{adversarial-recon}}
-->

where $\mathcal{L}_{\text{adversarial-recon}}$ is computed similarly to <ref>eq:adv_recon</ref>, but using the post-hoc causal importances and the targeted reconstruction loss. There is also one additional constraint imposed on the adversarial sampler compared to VPD base training: Only alive subcomponents on the prompt have their masks adversarially optimized, other subcomponents have their masks drawn stochastically. This is because we want to prevent the adversary from finetuning on data dependent noise inside the many inactive components of the model, see <ref>sec:vpd_methods-adv</ref>. In base training, this is accomplished by forcing the adversary to use the same $r^l_c$ for many data points. For post-hoc optimization we cannot do this, because we only have a single prompt available. But the causal importance function has already pre-filtered the subcomponents to exclude those that were not involved in computing the prompt at all, so we attempt to sidestep this issue by restricting the adversary to subcomponents that were alive on the original prompt.

##### Optimization procedure


We optimize the pre-sigmoid parameters $\phi^l_{t,c}$ using AdamW with a cosine learning rate schedule and brief linear warmup. The model weights and subcomponent parameters ($U$, $V$) are frozen throughout; only the post-hoc causal importance parameters are updated. The optimization typically converges within a few hundred steps, since it starts from a good initialization and optimizes over a single prompt rather than a dataset. The result is a set of refined causal importance values $\tilde{g}^l_{t,c}$ that are sparser than the base values: many subcomponents that were causally important for the full output are driven to zero importance when only a specific behavior is targeted. The surviving subcomponents—those with $\tilde{g}^l_{t,c} > 0$—form the nodes of the interaction graph for that behavior, and gradient attributions (<ref>app:gradient_attributions</ref>) are then computed between them.



### Nonlinear parameter subcomponent interactions {toc: Nonlinear subcomponent interactions}

<label id="app:interactions-gis-vs-coact"/>

In our case studies, we traced the relationships between subcomponent activations in particular computations using attributions. However, this is not a complete account of how the model computes its outputs. Attributions only measure how strongly one subcomponent activation influences another; they do not describe the actual functional relationship between them. To fully reverse engineer neural networks with VPD, we will need some account of how downstream subcomponent activations are actually computed from upstream ones.

For some matrices (such as MLP Up, query, key, and value projection matrices), this should not be difficult. The connections to their preceding subcomponent activations are linear (apart from the norms), so they can be understood almost entirely as linear combinations of preceding subcomponent activations.

For MLP Down projection and attention output matrices, however, nonlinearities in the computational graph separate them from preceding subcomponent activations: Neurons in the case of MLP Down projections, and attention heads in the case of attention output matrices. For MLP Down projection subcomponents in particular, every subcomponent activation is a linear combination of many MLP neuron activations, each of which is potentially a nonlinear function of all MLP Up matrix subcomponent activations.

One might therefore worry that the nonlinear interactions between MLP Up matrix subcomponent activations could be inherently very complicated. We cannot exclude this possibility at present, but there are some theoretical and empirical reasons to think that these interactions may be much simpler than the raw number of nonlinearities might suggest.

#### Theoretical argument

To the extent that the network implements different circuits in the same MLP—such as a lookup for which city the Eiffel Tower is in and a modular addition algorithm for the months of the year—it is actively incentivised to avoid nonlinear interactions between them. Otherwise, the circuits would interfere with each other, potentially producing wrong results. So, to the extent that two subcomponents parametrize two unrelated circuits, they should not interact much. From our interpretations of the subcomponents in <ref>sec:param-comps-interpretable</ref>, it appears that many of them are quite specialised to very different contexts, and thus presumably would not interact substantially. There are clusters of related subcomponents, such as those for bracket closing from our analysis in <ref>sec:case-studies-bracket</ref>, and these presumably could interact quite a bit. But smaller blocks of mutual interaction would still be much easier to analyze than a single block of all subcomponents interacting with all other subcomponents. There also appear to be some subcomponents, such as the nearly-always-active "biases," that would presumably interact nonlinearly with almost everything else. But characterising these interactions for a reasonably small number of subcomponents still seems quite feasible.

#### Preliminary empirical investigation

We can approximately measure the interaction strength between MLP Up projection matrix subcomponents at neurons. Specifically, we compute interaction matrices $I_{c,c'}$ that crudely measure two things: 

1. **Weight overlap**: How strongly different subcomponents $c, c'$ connect to the same neurons with sizeable weights, and
2. **Activation overlap**: How often they are causally important with large activations at the same batch and sequence index:

<!-- $$I^l_{c,c'}:=\frac{\left(\sum_{i} \vert U^l_{i,c}\vert \vert U^l_{i,c'}\vert\right)}{\left(\sum_i \vert U^l_{i,c}\vert^2\right)}\frac{\left(\sum_{b,t} \vert g^l_{b,t,c} a^l_{b,t,c}\vert  \vert g^l_{b,t,c'} a^l_{b,t,c'}\vert\right)}{\left(\sum_{b,t}\vert g^l_{b,t,c} a^l_{b,t,c}\vert^2\right)}$$ -->

```equation
tex:
  \htmlClass{hc-im-I}{I^l_{c,c'}}
  :=
  \htmlClass{hc-im-weight-overlap}{
    \frac{
      \left(
        \sum_{i}
        \htmlClass{hc-im-U-abs}{\vert U^l_{i,c}\vert}
        \htmlClass{hc-im-U-abs-prime}{\vert U^l_{i,c'}\vert}
      \right)
    }{
      \left(
        \sum_i \vert U^l_{i,c}\vert^2
      \right)
    }
  }
  \htmlClass{hc-im-act-overlap}{
    \frac{
      \left(
        \sum_{b,t}
        \htmlClass{hc-im-ga}{\vert g^l_{b,t,c} a^l_{b,t,c}\vert}
        \htmlClass{hc-im-ga-prime}{\vert g^l_{b,t,c'} a^l_{b,t,c'}\vert}
      \right)
    }{
      \left(
        \sum_{b,t}\vert g^l_{b,t,c} a^l_{b,t,c}\vert^2
      \right)
    }
  }
tips:
  - hc-im-I: Interaction strength of subcomponent c' on subcomponent c at layer l. Normalised so diagonal entries I_{c,c} = 1.
  - hc-im-weight-overlap: Weight overlap factor: Measures how strongly c and c' connect to the same neurons via the MLP input matrix
  - hc-im-U-abs: Absolute weight of subcomponent c at neuron i
  - hc-im-U-abs-prime: Absolute weight of subcomponent c' at neuron i
  - hc-im-act-overlap: Activation overlap factor: Measures how often c and c' are simultaneously active and causally important on the same inputs
  - hc-im-ga: Effective activation of subcomponent c: Causal importance g times subcomponent activation a
  - hc-im-ga-prime: Effective activation of subcomponent c': Causal importance g' times subcomponent activation a'
```

Intuitively, $I^l_{c,c'}$ measures how often subcomponents $c$ and $c'$ make a large contribution to the preactivations of the same MLP neurons $i$ at the same batch and sequence indices $b, t$. If the contribution of subcomponent $c$ to the preactivation of neuron $i$ is much larger in magnitude than the corresponding contribution of subcomponent $c'$ (i.e. $U^l_{i,c} g^l_{b,t,c} a^l_{b,t,c} \gg U^l_{i,c'} g^l_{b,t,c'} a^l_{b,t,c'}$), then $c'$ will be mostly irrelevant for determining the nonlinear response of neuron $i$ on that data point.

The matrix is normalised such that all diagonal entries equal $1.0$, and each row can be read as estimating the interaction strength between subcomponent $c$ and other subcomponents $c'$, relative to the self-interaction of $c$. If an off-diagonal entry $I_{c,c'}$ is much smaller than $1.0$, this indicates that subcomponent $c'$ does not substantially interact with subcomponent $c$ at the neurons. <footnote>Note that $I$ is not symmetric: $I_{c,c'}$ and $I_{c',c}$ can differ. This is intentional. If subcomponent $c$ influences a neuron's preactivation much more strongly than subcomponent $c'$, the computational pathway of the latter is likely heavily influenced by the former, but not vice versa.</footnote>

We expect some off-diagonal entries to be large. For example, for subcomponents that form part of the same component. But generally speaking, the fewer large off-diagonal entries there are, the easier it should be to describe the computation in an MLP in terms of components without considering many inter-component interactions.

<figure>
<label id="fig:I_h"/>
<img src="figures/I_h_0_mlp_c_fc.png">
<figcaption>Entries of the interaction matrix $I_{c,c'}$ for the layer $0$ MLP up projection matrix subcomponents. Entries greater than $1.0$ are clamped to $1$. Indices are sorted by the components that each subcomponent belong to. Entries $\geq 1.0$ in a row indicate that the nonlinear interaction between subcomponents $c'$ and $c$ is large compared to the self-interaction of $c$ on the diagonal. Many interactions are either block-diagonal, indicating they take place inside higher-rank components, or arranged along vertical and horizontal lines, indicating they are caused by a relatively small number of highly interacting components.</figcaption>
</figure>

<!--DAN: I don't like the title of this figure. First because it says c_fc which people won't know, and also it doesn't say what exactly are alive. Probably fine to just remove the (1365 alive) and mention it in the caption  -->

<figure>
<label id="fig:I_dist"/>
<img src="figures/I_dist_h_0_mlp_c_fc.png">
<figcaption>Histogram of matrix entries of the interaction matrix $I_{c,c'}$ for the layer $0$ up projection components. Most entries are much smaller than the self-interaction $1.0$.</figcaption>
</figure>

<ref>fig:I_h</ref> shows the $I$ matrix entries for the layer 0 MLP input matrix, with the subcomponent indices sorted by the cluster components they belong to.  <ref>fig:I_dist</ref> shows a histograms of the $I$ entries. We can see that while there are certainly quite a few large off-diagonal entries, many of them represent interactions within components (the diagonal blocks), or interactions of a small set of highly interacting subcomponents with all others (the vertical and horizontal stripes). Plots for the interaction matrices of other layers can be found in the next subsection. We stress that this is a very prelimary investigation and the matrices $I_{c,c'}$ are a crude and impresice measure of nonlinear interactivity in many ways. 

For example, they do not quantify how much particular nonlinear interactions between subcomponents actually influence downstream observables like the model output.


Ultimately, what will matter in practice is whether we can use parameter components to interpret nonlinear interactions well enough to reverse engineer the algorithms neural networks have learned to implement. While we may have some reasons for optimism on this question, we cannot provide a real answer to it yet.



#### Interaction matrix plots for all MLPs {toc: Interaction plots for all MLPs}

<label id="app:non_linear_plots"/>

Here, we show raw heatmaps and histograms for the interaction matrices $I_{c,c'}$ quantifying nonlinear interactions between MLP up projection matrix subcomponents at neurons for the other three mlp layers. For the heatmaps, indices are sorted by the components subcomponents belong to. Entries $\geq 1.0$ in a row indicate that the nonlinear interaction between subcomponents $c'$ and $c$ is large compared to the self-interaction of $c$ on the diagonal. Many interactions are either bock-diagonal, indicating they take place inside highe rank components, or arranged along vertical and horizontal lines, indicating they are caused by a relatively small number of highly interacting components. The layer 1 mlp has particularly many interactions, which may be an additional indicator that the VPD decomposition of this transformer layer is somewhat pathological.

 <!-- See <ref>app:interactions-gis-vs-coact</ref> for context on the definition of $I_{c,c'}$.-->


 <!-- 
<figure>
<img src="figures/I_h_0_mlp_c_fc.png">
<figcaption>Entries of the interaction matrix $I_{c,c'}$ for the layer 0 MLP up projection subcomponents. Indices are sorted by the components subcomponents belong to.</figcaption>
</figure>

<figure>
<img src="figures/I_dist_h_0_mlp_c_fc.png">
<figcaption>Histogram of the entries of the interaction matrix $I_{c,c'}$ for the layer 0 MLP up projection subcomponents.</figcaption>
</figure>
-->
<figure>
<img src="figures/I_h_1_mlp_c_fc.png">
<figcaption>Entries of the interaction matrix $I_{c,c'}$ for the layer 1 MLP up projection subcomponents. Indices are sorted by the components subcomponents belong to.</figcaption>
</figure>

<figure>
<img src="figures/I_dist_h_1_mlp_c_fc.png">
<figcaption>Histogram of the entries of the interaction matrix $I_{c,c'}$ for the layer 1 MLP up projection subcomponents.</figcaption>
</figure>

<figure>
<img src="figures/I_h_2_mlp_c_fc.png">
<figcaption>Entries of the interaction matrix $I_{c,c'}$ for the layer 2 MLP up projection subcomponents. Indices are sorted by the components subcomponents belong to.</figcaption>
</figure>

<figure>
<img src="figures/I_dist_h_2_mlp_c_fc.png">
<figcaption>Histogram of the entries of the interaction matrix $I_{c,c'}$ for the layer 2 MLP up projection subcomponents.</figcaption>
</figure>

<figure>
<img src="figures/I_h_3_mlp_c_fc.png">
<figcaption>Entries of the interaction matrix $I_{c,c'}$ for the layer 3 MLP up projection subcomponents. Indices are sorted by the components subcomponents belong to.</figcaption>
</figure>

<figure>
<img src="figures/I_dist_h_3_mlp_c_fc.png">
<figcaption>Histogram of the entries of the interaction matrix $I_{c,c'}$ for the layer 3 MLP up projection subcomponents.</figcaption>
</figure>


### Frequency minimality loss information theory motivation {toc: Frequency minimality loss motivation}

<label id="app:frequency_penalty_motivation"/>

Here, we provide an information theoretic motivation for the functional form of $\mathcal{L}_{\text{frequency-minimality}}$ based on minimising description length per data point: In a fixed dictionary of subcomponents, subcomponents that are more frequently causally important effectively need to be specified to more bits of precision to reconstruct the model's outputs.


In the idealized setting, subcomponents are vectors of real numbers. In reality, we instead store them as vectors of finite precision floats. This quantisation effectively induces a discrepancy $\delta^l_c$ in parameter space between the ideal parameter vector for subcomponent $c$ in matrix $l$, and our floating point approximation of it. At sufficiently high float precision, the expected size of this discrepancy will scale as $\approx a_1 2^{-b^l_c}$, where $b^l_c$ is a bit count and $a_1$ is some constant.
Suppose we want to keep the impact of this discrepancy on our decomposition low. Specifically, we want the number of bits $b^l_c$ to be large enough for the KL divergence between the VPD forward pass outputs and the target model forward pass outputs summed over the batch to stay below some fixed $\epsilon>0$. How large will we need to make $b^l_c$ as a function of $\epsilon$ to achieve this?

Over a batch of $B$ inputs of sequence length $T$, a subcomponent will be causally important with some frequency $f^l_c:=\frac{\sum^{B,T}_{b,t=1}\vert g^l_{b,t,c}\vert^0}{B T}$. For simplicity, we assume that applying some small perturbation of size $\delta$ along the direction of a subcomponent in parameter space does not change the model output at all on data points where $g^l_{b,t,c}=0$, but increases the KL divergence to the original model outputs by some $p(\delta)$ on data points where $g^l_{b,t,c}=1$, where $p$ is an analytic function that is approximately the same for every subcomponent and every data point. Then, the increase to the total loss summed over all $B T$ data points from adding a perturbation $\delta$ to subcomponent $c$ is of approximate size $\approx \sum^{B,T}_{b,t=1}\vert g^l_{b,t,c}\vert^0 p(\delta)$. This yields the inequality

$$\begin{aligned}
&\log_2(p(\delta))+\log_2(\sum^{B}_{b=1}\sum^{T}_{t=1}\vert g^l_{b,t,c}\vert^0)<\log_2(\epsilon)\\
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
&\log_2(h(\delta))+\log_2(\sum_{x,t}\vert g^l_c(x,t)\vert^0)<\log_2(\epsilon)\\
\end{aligned}
-->


Since $p$ is an analytic function, for sufficiently small $\delta$, it can be Taylor approximated to leading order as $a_2 \delta^n$ with some $n\in\{1,2,\dots\}$. Inserting this approximation yields:

$$\begin{aligned}
b^l_c&>\frac{1}{n}\log_2(\sum^{B}_{b=1}\sum^{T}_{t=1}\vert g^l_{b,t,c}\vert^0)-\frac{\log_2(\epsilon)}{n }+\frac{\log_2(a_2)}{n}+\log_2(a_1)\\
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
b^l_c&>\frac{1}{n}\log_2(\sum_{x,t}\vert g^l_c(x,t)\vert^0)-\frac{\log_2(\epsilon)}{n }+\frac{\log_2(a_2)}{n}+\log_2(a_1)\\
\end{aligned}
-->


So, the required bit precision $b^l_c$ for the parameters of a subcomponent grows approximately linearly with the logarithm of that subcomponent's number of causal importance activations across the dataset $\log_2(f^l_c)$. If we use a fixed dictionary of subcomponents to describe how the model computes its outputs, the mechanistic description length of our descriptions summed over a batch will thus have a term that scales as $\approx \sum^L_{l=1} \sum^C_{c=1} f^l_c \log_2(f^l_c)$. Substituting the definition $f^l_c=\frac{\sum^{B,T}_{b,t=1}\vert g^l_{b,t,c}\vert^0}{B T}$ and absorbing the $-\log_2(BT)$ term into the importance minimality loss yields
$$\begin{aligned}
\mathcal{L}_{\text{frequency-minimality}}=\frac{1}{BT}\sum^L_{l=1}\sum^B_{b'=1}\sum^T_{t'=1}\sum^C_{c=1}\vert g^l_{b',t',c}\vert^p \log_2(1+\sum^B_{b=1}\sum^T_{t=1} \vert g^l_{b,t,c}\vert^p)\,.
\end{aligned}$$

### Training Details and Hyperparameters

<label id="app:training-details"/>

**Target model training.**

Target model training artifacts can be found on WandB (<a href="https://wandb.ai/goodfire/spd/runs/t-9d2b8f02/files/final_config.yaml" target="_blank">config</a>, <a href="https://wandb.ai/goodfire/spd/runs/t-9d2b8f02/files/model_step_99999.pt" target="_blank">checkpoint</a>, <a href="https://wandb.ai/goodfire/spd/runs/t-9d2b8f02" target="_blank">run logs</a>).

The target model architecture is described in <ref>sec:langauge-model-details</ref> and <ref>tab:model-hyperparams</ref>.
It was trained on a subset of The Pile <cite>gao2020pile</cite> for $100,000$ steps with batch size $1024$ and context length $512$.
We used Adam <cite>kingma2017adam</cite> with learning rate $3 \times 10^{-4}$ (cosine decay to $10\%$), weight decay $0.1$, gradient clipping at $1.0$, and $600$ warmup steps.
Training used `bfloat16` mixed precision and `torch.compile`.

**VPD training.**

Decomposition artifacts can be found on WandB (<a href="https://wandb.ai/goodfire/spd/runs/s-55ea3f9b/files/final_config.yaml" target="_blank">config</a>, <a href="https://wandb.ai/goodfire/spd/runs/s-55ea3f9b/files/model_400000.pt" target="_blank">checkpoint</a>, <a href="https://wandb.ai/goodfire/spd/runs/s-55ea3f9b" target="_blank">run logs</a>).

VPD decomposes 24 weight matrices (6 per layer: `c_fc`, `down_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`) into rank-one subcomponents with delta components enabled.
Training ran for $400,000$ steps with batch size $64$ on the same Pile dataset with context length $512$.
The $U,V$ and CI function parameters were jointly optimized with AdamW (weight decay $0$), initial learning rate $5 \times 10^{-5}$ with cosine decay to $10\%$ of the initial value.
$U,V$ gradients were clipped at norm $0.01$.
One stochastic mask sample ($S=1$) was drawn per step.
Faithfulness warmup ran for $400$ steps (AdamW, lr $= 10^{-3}$, weight decay $0$), optimizing only the $U,V$ parameters against $\mathcal{L}_{\text{Delta-L2}}$ before the main training loop.
The output divergence measure $D$ is KL divergence throughout.

The causal importance function $\Gamma$ is a shared bidirectional transformer (architecture described in <ref>tab:ci-hyperparams</ref>).
It takes RMS-normalized concatenations of all 24 pre-weight activations (total input dimension $D = 27,648$) and outputs $C_{\mathrm{total}} = 39,936$ causal importance values via a leaky hard sigmoid ($\alpha = 0.01$).

The $p$-norm exponent in both $\mathcal{L}_{\mathrm{importance\text{-}minimality}}$ and $\mathcal{L}_{\mathrm{frequency\text{-}minimality}}$ is linearly annealed from $p_0 = 2.0$ to $p_{\mathrm{final}} = 0.4$ over the full training run.

**Adversarial reconstruction.**

To optimizes the persistent sources in the persistent PGD adversarial loss, an Adam optimizer was used with $\beta_1 = 0.5$, $\beta_2 = 0.99$, learning rate $0.01$ (constant schedule with $2.5\%$ warmup).
Sources are scoped per batch element per sequence position (i.e. each individual batch element and sequence position has its own source), and each source receives $2$ warmup PGD steps per training step before the final loss computation.
Stochastic and adversarial reconstruction losses both use uniform-$k$-subset routing, where a random subset of the 24 weight matrices is masked on each step.

**Combined minimality loss in code.**

For efficiency, in the training code $\mathcal{L}_{\mathrm{importance\text{-}minimality}}$ and $\mathcal{L}_{\mathrm{frequency\text{-}minimality}}$ are implemented as a single fused term per layer, which factors their shared per-component sum:

$$
\mathcal{L}_{\mathrm{minimality}} \;=\; \frac{1}{BT} \sum^{L}_{l=1} \sum^{C}_{c=1} \left[\, s^l_c \;+\; \beta \, s^l_c \, \log_2\!\left(1 + s^l_c\right) \right],
\qquad
s^l_c \;=\; \sum^{B}_{b=1} \sum^{T}_{t=1} \vert g^l_{b,t,c} + \epsilon\vert^{p}.
$$

Here $\beta = 0.5$ is the frequency minimalty weight and $\epsilon$ is a small constant for numerical stability. This is functionally equivalent to
summing $\mathcal{L}_{\mathrm{importance\text{-}minimality}} + \beta \,\mathcal{L}_{\mathrm{frequency\text{-}minimality}}$ as defined in <ref>eq:minimal</ref> and <ref>eq:freq_minimality</ref>; we fuse them because both terms depend on the same per-component sum $s^l_c$, so the fused form avoids recomputing it.

**Loss terms and coefficients.**

<ref>tab:vpd-loss-coefficients</ref> lists all loss terms and their coefficients.

<label id="tab:vpd-loss-coefficients"/>
| **Loss term** | **Reference** | **Coefficient** |
|---|---|---|
| $\mathcal{L}_{\text{Delta-L2}}$ (auxiliary loss; a.k.a. parameter-faithfulness) | <ref>eq:delta_l2</ref> | $10^{7}$ |
| $\mathcal{L}_{\mathrm{stochastic\text{-}recon\text{-}subset}}$ (stochastic KL) | <ref>eq:random_recon</ref> | $0.5$ |
| $\mathcal{L}_{\mathrm{adversarial\text{-}recon\text{-}subset}}$ (persistent PGD KL) | <ref>eq:adv_recon</ref> | $0.5$ |
| $\mathcal{L}_{\mathrm{importance\text{-}minimality}}$ ($\ell_p$ on CI values) | <ref>eq:minimal</ref> | $2 \times 10^{-4}$ |
| $\mathcal{L}_{\mathrm{frequency\text{-}minimality}}$ (superlinear CI frequency penalty) | <ref>eq:freq_minimality</ref> | $1 \times 10^{-4}$ |
*VPD loss terms and their coefficients. The importance minimality loss uses $p$-annealing from $2.0$ to $0.4$. In practice we implement $\mathcal{L}_{\mathrm{importance\text{-}minimality}}$ and $\mathcal{L}_{\mathrm{frequency\text{-}minimality}}$ as a single fused term with an inner weight $\beta = 0.5$ on the frequency component (see above); this is functionally equivalent to the two losses with the coefficients shown. All reconstruction losses use KL divergence.*

**Subcomponent counts.**

<ref>tab:vpd-subcomponent-counts</ref> lists the number of rank-one subcomponents $C$ we give to each module at initialization.

<label id="tab:vpd-subcomponent-counts"/>
| **Module type** | **Subcomponents ($C$) per layer** |
|---|---|
| `c_fc` (MLP up-projection, $768 \times 3072$) | 3072 |
| `down_proj` (MLP down-projection, $3072 \times 768$) | 3584 |
| `q_proj` (query projection, $768 \times 768$) | 512 |
| `k_proj` (key projection, $768 \times 768$) | 512 |
| `v_proj` (value projection, $768 \times 768$) | 1024 |
| `o_proj` (output projection, $768 \times 768$) | 1024 |
| **Total per layer** | 9984 |
| **Total (4 layers)** | 39,936 |
*Number of rank-one subcomponents per module type at initialization.*

The CI function architecture is shown in <ref>tab:ci-hyperparams</ref>.

The training and evaluation losses achieved by the primary training run studied in this paper are listed in <ref>tab:vpd-eval-losses</ref> and <ref>tab:vpd-train-losses</ref> respectively.

<label id="tab:vpd-train-losses"/>
| Loss | Value |
|---|---|
| Total | $24.62$ |
| Delta-L2 (mse) | $0.00000240$ |
| StochasticReconSubsetLoss (KL) | $0.2419$ |
| PersistentPGDReconLoss (KL) | $0.5733$ |
| ImportanceMinimalityLoss | $1102.0$ |
*Training losses (Measured at final step).*

<label id="tab:vpd-eval-losses"/>
| Loss | Value |
|---|---|
| StochasticReconSubsetLoss (KL) | $0.2381$ |
| PGDReconLoss (KL) | $0.9268$ |
| StochasticHiddenActsReconLoss (mse) | $0.4130$ |
| CIHiddenActsReconLoss (mse) | $0.8464$ |
*Evaluation reconstruction losses.*


**CLT/PLT WandB links**

| Used in | What | WandB link |
|---|---|---|
| <ref>fig:pareto-mse</ref>; <ref>fig:splitting-heatmap</ref> | PLT/CLT, local-MSE objective, $k \in \{8, 16, 32, 64\}$ | <a href="https://wandb.ai/mats-sprint/pile_local_sweep_jose" target="_blank">dict_4k</a>, <a href="https://wandb.ai/mats-sprint/pile_local_sweep_jose_32k" target="_blank">dict_32k</a> |
| <ref>fig:pareto-e2e</ref> | PLT/CLT, end-to-end KL objective, $k \in \{8, 16, 32, 64\}$, three training modes (`cascading` = error-propagating, `parallel` = clean-input, `independent` = single-layer) | <a href="https://wandb.ai/mats-sprint/pile_e2e_sweep_jose" target="_blank">dict_4k</a>, <a href="https://wandb.ai/mats-sprint/pile_e2e_sweep_jose_32k" target="_blank">dict_32k</a> |
| <ref>tab:seed-mmcs</ref> — PLT/CLT seed runs | 5 seeds $\times$ {PLT local-MSE, PLT e2e-independent, CLT local-MSE, CLT e2e-parallel}, $k = 16$, 4k dict | <a href="https://wandb.ai/mats-sprint/pile_multiseed_jose2" target="_blank">multiseed</a> |
| <ref>tab:seed-mmcs</ref> — VPD seed runs | 5 VPD seed runs (otherwise identical to the main decomposition) | <a href="https://wandb.ai/goodfire/spd?nw=n9l0amrrudc" target="_blank">VPD multiseed</a> |
| <ref>tab:seed-mmcs</ref> — hidden-activation aux-loss VPD | VPD trained with an auxiliary stochastic-forward-pass hidden-activation MSE loss | <a href="https://wandb.ai/goodfire/spd/runs/s-aa4fec0a" target="_blank">VPD hidden-act run</a> |
| <ref>fig:feature_splitting</ref>; <ref>fig:splitting-heatmap</ref> | VPD capacity sweep ($0.5\times$, $1\times$, $2\times$, $4\times$ subcomponents); $1\times$ is the main run above | <a href="https://wandb.ai/goodfire/spd/workspace?nw=ckmtpmd21yl" target="_blank">capacity_sweep</a> |
| <ref>fig:adv-vs-no-adv</ref> | No-adversarial-loss control run; otherwise identical training configuration to the main decomposition | <a href="https://wandb.ai/goodfire/spd/runs/s-05ef623e" target="_blank">VPD no-adversarial-loss run</a> |

All activation-based comparison runs target the same <a href="https://wandb.ai/goodfire/spd/runs/t-9d2b8f02" target="_blank">t-9d2b8f02</a> model and use $\text{LR} = 3 \times 10^{-4}$, batch size $4096$, sequence length $512$, $500$M tokens of the Pile, with BatchTopK activation.



<!-- Graveyard -->


<!-- Graveyard: VPD base training yields a set of components which sum to the target model weights, and a causal importance function that tries to predict which components can be ablated without changing the final output of the model on any particular data point. To understand the target model's behavior, we also make use of additional tools: We further prune the number of components on particular prompts down to only those involved in some particular behavior we are interested in by optimizing new causal importances to reconstruct only some aspects of the target model's output. We also compute gradient attributions between components to obtain interaction graphs that visualize how components interact with each other on the forward pass. 
**Gradient attributions**

We compute gradient attributions txdo cite between pairs of causally important components in adjacent layers. Many works have pointed out issues that can cause gradient attributions to be unfaithful, such as saturated softmax functions in attention layers. We use them here merely as a supplementary tool to identify some qualitative relationships between components. For more details on the gradient attributions, see <ref>app:gradient_attributions</ref>.




circuits intro graveyard

<!-- TODO(Lee)(High priority): Figure out how to restructure this wrt to discussion section content. Basically most of this section's intro feels like-->

<!-- At this point, it's worth pausing to reflect on what our parameter decomposition approach has actually bought us with regard to the goals of mechanistic interpretability. We view the primary goal of mechanistic interpretability as understanding a neural networks computational graph. -->

<!--Parameter components are quite a different kind of object from a transcoder latent. They are not simple thresholded linear functions like in PLTs or CLTs. They do not simply 'read' from a single activation direction and 'write' to another. Instead, they are vectors in parameter space (albeit ones that are 'used' by the network only sparsely). 
The causally important in particular activation manifolds (i.e. regions of activation space), rather than particular activation directions. As we have seen, these regions of activation space nonetheless tend to share particular semantic properties, suggesting a shared underlying computational similarity.-->
<!--DAN (resolved): rank 1 subcomponents do only read from a single direction rather than a manifold. Agree that a full component can read from a manifold, but the "as we have seen" line is confusing because we've only looked at single subcomponents prior to this. I'd also make more explicit that components read from manifolds but single subcomponents read from directions, as I can imagine people being confused here. 
LEE: This is a matter of perspective. You can either view a subcomponent as reading from a direction and writing to a line (1d manifold) or vice versa. It just whichever one you decide to hold constant. This of course leaves aside their interactions, which makes them able to be involved in computing high dim manifolds. --- I've modified the text a bit to make it clear I'm talking about CI not component activation.
-->
<!--Lucius: I agree with Dan that this seems confusingly worded/false as is. I'm still not quite sure what it's trying to say.  -->

<!--This difference highlights one of the key distinctions between sparse dictionary learning and parameter decomoposition methods. Dictionary learning methods are not 'algorithmically neutral'<footnote>Full credit to our colleague Owen Lewis for this framing and the framing we use elsewhere in this section.</footnote>. CLTs constrain their descriptions of neural computation to use a particular type of nonlinear computation (thresholded linear functions), but this may not be a nonlinearity that preserves whatever systematic regularity that was present in the computations of the target model. Dictionary learning methods replace the target model's direct acyclic graph with a sparser graph that uses thresholded linear functions in its nodes and scalar valued edges. This is a design choice with many attractive properties. But we should be open to the possibility that the hypothesis class of CLT graphs may not be rich enough to represent the hypotheses for parsimonious explanations of how neural networks compute their behavior.


Parameter decomposition methods, on the other hand, *are* algorithmically neutral. The hypothesis class of parameter decompositions can certainly represent what neural networks have learned, since they are essentially the same class! But they pay for this neutrality by permitting more complex interactions than linear thresholded functions. They therefore permit potentially complex interactions between parameter components.

To use parameter decomposition for interpretability, we need a way to study how information flows between parameter components. But if interactions are potentially complex, how should we do this? We solve this problem in a similar way to CLTs: Attribution graphs. We use attributions to measure the strength of the interaction between causally important subcomponents. In particular, we use gradient attributions, though we use stop-gradients to measure only the 'direct' effects of one subcomponent on another (<ref>sec:attr-calcs</ref>). Using gradients in this way 'abstracts away' the complexity of these interactions, summarizing into a single number interactions that may pass simultaneously through many nonlinearities, thus achieving one of the same goals as CLT attribution graphs. 

However, attributions are only 'local' measures of interaction strength; their strength depends on the particular datapoint that we measure them on. Many works have pointed out issues (such as saturated softmax functions in attention layers) that can cause such local attributions to be unrepresentative of more 'global' measures <cite>kramár2024atpefficientscalablemethod, jafari2025relpfaithfulefficientcircuit</cite>. In order to identify more 'global' measures of interaction strength, we would need to better characterize the nonlinear relationships between parameter subcomponents. This is an important research priority, and one that we've already begun exploring, but not something that this paper covers in detail. We do nonetheless provide analysis that suggests parameter subcomponents of MLP matrices, despite not being selected to have simple interactions, tend toward it anyway (<ref>app:interactions-gis-vs-coact</ref>).-->
<!--DAN: (Lucius) This last bit tends to read like an incidental finding. It would be nice (but not crucial) to say why CiS gives us reason to expect this to be the case. -->

<!--Using attribution graphs lets us tell interpretability stories about what information is flowing within the network, in much the same way as CLTs do. For now, we abstract away the complex interactions between parameter subcomponents using attributions. CLTs abstract away the complex interactions into linear interactions between thresholded linear latents, which is perhaps too simple an approximate to permit parsimonious explanations of the underlying computation. If both are abstractions, then an important question is "are they faithful to the target model's mechanisms?" While VPD optimizes for mechanistic faithfulness directly, it remains unclear how mechanistically faithful CLTs are <cite>ameisen2025circuit, lange2026crosslayer</cite>.-->





<!-- 

conclusion graveyard:

The ability to reverse engineer neural algorithms in terms of their parameters suggests that we could understand what models are learning during training, helping us steer the training process so that the model learns to have more of the properties that we want, and less of the properties that we do not. 

decompose and analyze parameter components hints at a ability to understand 

reverse engineer entire neural networks in terms of their parameters. This might let us rewrite whole models to have more of the properties we want, and less of the properties we do not. 

Reflecting on the long term trajectory of mechanistic interpretability, we think parameter decomposition may open up new affordances

By decomposing neural network parameters into 'chunks with shared causal responsibility',  it breaks apart 

into these 'monoliths', revealing endless computational forms, most beautiful and most wonderful. 

blocks though the computations that they implement are undecomposable. that (with the nonlinearities) implement . We believe that VPD offers a way to decompose these hitherto inscrutable matrices into chunks that are each responsible for separable computations, at least where that separation is possible.  represents an important step toward decomposing neural networks into parts




 -->





<!-- Contributions graveyard -->
<!--Collecting rough notes below

**Please add things as you think of them. It's fine if it's a mess: It's fine to structure by 'content' or 'individual' or both. I've added some examples.**

Heuristics: Err on the side of overselling your contributions and please let Lee and others collectively take responsiblity for ensuring their contributions have been fairly represented. We'll work together to make sure everyone is happy with this!


- Lucius

  - Conceptualised adversarial reconstruction loss and its implementation via PGD on sources, with some input from Linda. Did early method and hyperparameter iteration to get adversarial losses working on toy models and the simple stories model.
  - Based on some discussions with Linda and external folks as well as empiricial iteration, conceptualized the frequency-minimality loss. Also did most of the testing and tuning for it.
  - Conceptualized delta components and did the early testing for them.
  - Did a lot of vpd hyperparameter tuning in general. So did Oli and Dan.
  - Conceptualized the new lower-leaky sigmoid after discussion with Lee, except for the sign exception on the straight-through estimator, which was conceptualised by Linda after Lucius noticed a problem with the previous version.
  - Conceptualised post-hoc causal importance optimization and post-hoc adversarial optimization restricted to base graph nodes. Did ?most? of the hyperparameter tuning for post-hoc causal importances.
  - Conceptualised first form of the clustering algorithm (mdl framing, initial mdl loss function, hierarchical merging, stopping based on mdl minimum, picking alpha based on coactivation threshold, ...), did some of the empirical iteration to pick a clustering for the paper
  - Did a lot of the conceptualisation work for the attributions we use (gradient stopping, etc.) with input from Oli, Dan and Lee.
  - early conceptualisation for the nonlinear interaction metric along with Lee (IIRC the current interaction metric was proposed by me, but could be wrong on that), nonlinear interaction experiments for the paper
  - Did the first biostory on the simple stories model and two of the biostories in this paper.
  - some early conceptualisation for model editing, helped Oli with the model editing experiment in the paper
  - conceptualised using component activations on top of causal importances for interp.
  - initial drafts for some sections in the paper (some I remember: two biostories, methods frequency penalty, methods mechanistic faithfulness, methods adversarial loss, nonlinear interactions, model editing, parts of the discussion section, training recipe, most of the mathy sections in the appendix).
- Oli
  - was the primary responsible person for the visualization app we used internally, and for many of the interactive versions of them in paper.
  - Came up with using persistence in adversarial training loss and did a lot of hyperparameter optimization for it
  - Came up with our current causal importance function architecture, also came up with shared_mlp and global_shared_mlp causal importance function architecture we were using for a long time before that, and the vector gate mlp before that # oli here - "came up with" is generous here - it's definitely not very clever
  - Main responsible for autointerp and intruder detection comparisons
  - (unfruitful) experimentation with binarized CI values
  - Main responsible for making the paper look nice, with some contributions from Lee (and others?), in collaboration with others.
  - Managed codebase alongside Dan
    - <putting eng stuff here, feel free to ignore. most shared with Dan>
    - multi-gpu and multi-node stuff
    - implementation of core SPD library
    - routing stuff
  - optimizations of clustering using sparse representations (went from 500k to 10m token clustering runs)
  - diagnosed clustering failures and came up with the rank-exponential sampler to overcome these
  - early explorations of model editing, later did the final version with Lucius
  - developed tooling for background agents (tbh I don't know how productive this was. Lucius I think used it a bit though?)
  - can't remember exactly how it was split up but Dan and I did the PGD implementations
- Lee
  - Did an initial draft of some sections
  - Planned paper
  - Did an initial implementation of the global causal importance function
  - Overall management of the project.
  - Main point of contact for misha and nathan and bart and gave input on their work throughout the mats program (so did lucius)
  - Did the previous token behavior analysis.
  - Did first Pile training run? Dan did this?
  - Did geometric consistency seed analysis.
  - Various didactic figures used in the paper

- Misha
  - Clustering - developed algorithm, with inputs from Nathan, Lucius, Lee, and wrote up an initial draft of the paper section on clustering. Oli (and Dan?) made some optimizations to the clustering algorithm that helped it scale.
- Dan
  - Implemented and tuned hyperparameters for various methods that the team had over the length of the project (the ideas typically did not originate from me).
  - Conceptualised the part of the current adversarial loss which does several steps of warmup of the persistent sources for each outer loss step.
  - Model pretraining
  - Managed codebase alongside Oli
  - Created more efficient clustering implementation
  - Various contributions to the internal visualisation app and the attribution graph visualisation in this paper (Oli was the main person here)
- Nathan
  - Came up with idea for subset routing # (oli): I think this was Nathan (NH - someone else also mentioned it a while back but I had the first few experiments with it)
  - P-annealing (and a bunch of other methods optimizations on resid 3 mlp that didn't work)
  - help misha on clustering
  - (probably not relevant with how paper is shaping up) toy evals metrics and toy models of modular arithmetic 

- Bart
  - Trained the per-layer and cross-layer transcoders used for comparisons to VPD
  - Did the evaluation and analyzes of the reconstruction performance comparing VPD to transcoders
  - Drafted the section comparing VPD to transcoders
  - Did the feature splitting analysis and drafted this section


- etc. etc. please add your own messy notes above!-->