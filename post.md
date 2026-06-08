---
title: Interpreting Language Model Parameters
authors:
  - name: TODO authors
    url: https://goodfire.ai
affiliation: TODO affiliations
correspondence: tbd@tbd.com
published: "2026"
---

TODO abstract

## Introduction

<!-- Structure in the parameters of language models is responsible for their remarkable intelligence. The trainable parameters of these neural networks, in interaction with the architecture and dataset, learn to implement algorithms that we do not know how to design directly. -->

Language models are remarkably intelligent. During training, their parameters learn to implement neural algorithms that we do not know how to design directly.

We can thus train machines to solve tasks that otherwise resist engineering solutions, incidentally creating objects that are of great scientific interest in their own right. However, since we did not design these neural algorithms ourselves, it means that an increasing portion of our daily lives depend on increasingly capable systems that we do not deeply understand <cite>Bengio2026InternationalAISafety</cite>.

Mechanistic interpretability aims to reverse engineer neural networks so that we can understand networks such as language models. But reverse engineering requires decomposing a system into simpler parts that we can study in relative isolation. A key barrier to reverse engineering neural networks is that it is not obvious how best to decompose them into such parts <cite>mueller2024questrightmediatorhistory, sharkey2025openproblemsmechanisticinterpretability</cite>.
Unfortunately, naive choices of these parts—such as neurons, attention heads, or whole layers—don't always map to individual, interpretable computations <cite>hinton1981parallel, wei2015understandingintraclassknowledgeinside, nguyen2016multifacetedfeaturevisualizationuncovering, olah2017feature, janiak2023polysemantic, jermyn2023attention, yun2021sparse, lindsey2024crosscoders</cite>.

Alternative approaches to decomposition, such as transcoders <cite>dunefsky2024transcodersinterpretablellmfeature, ameisen2025circuit</cite> or mixtures of linear transforms <cite>oldfield2025towards, lindsey2025molts</cite>, typically involve fitting a set of simple functions to the transitions between activations at different layers in the network, and linearly combining the outputs of these simple functions.
The idea here is to approximate the complex, nonlinear function implemented by the network's layers using a simpler, easier to understand function. These methods, sometimes called *activation-based decomposition* methods, have led to significant advances in our understanding of the intermediate representations inside neural networks when computing their outputs <cite>dunefsky2024transcodersinterpretablellmfeature, ameisen2025circuit</cite>.

But identifying representations is not the same as understanding the computations that use those representations as their inputs and outputs. Unfortunately, because the simpler functions that these methods use are of a different functional form to the original network, it is hard to relate their accounts of network function to the actual objects that are doing the computations—the network's parameters and its nonlinearities.

This is not just a theoretical issue; it prevents us from achieving practical engineering goals. For example, it makes it challenging to know how to make edits to a model's parameters that change its neural algorithm in a predictable, desirable way (while also avoiding unpredictable side effects). It also makes it hard to predict how the model's neural algorithm will perform on a different distribution than the one on which it was studied. 

The mismatch of functional form between models and their activation-based decompositions decompositions is an important issue, but it is not the only one: Activation-based methods have not yet yielded decompositions that exhibit a fully satisfactory level of mechanistic faithfulness <cite>ameisen2025circuit</cite>, and suffer from a number of other issues (See <cite>sharkey2025openproblemsmechanisticinterpretability</cite> for review).

<!-- Consequently, methods such as sparse dictionary learning \cite{yun2021sparse, Sharkey_2022, cunningham2023sparse, cunningham2023sparseautoencodershighlyinterpretable, bricken2023monosemanticity}, transcoders \cite{dunefsky2024transcodersinterpretablellmfeature, ameisen2025circuit}, and mixtures of linear transforms (MOLTs) \cite{oldfield2025towards, lindsey2025molts} were introduced to decompose datasets of neural activations, with the hope that they would identify units that approximate the network's underlying computational units. These methods, sometimes called \textit{activation-based decomposition} methods, unfortunately suffer from a range of issues, including feature splitting \cite{bricken2023monosemanticity, chanin2024absorptionstudyingfeaturesplitting} and unreliable level of mechanistic faithfulness \cite{ameisen2025circuit} (See \cite{sharkey2025openproblemsmechanisticinterpretability} for a more comprehensive review of these methods). Mechanistic unfaithfulness is suboptimal for activation-based methods' use in mechanistic analysis, and arises in part because these methods do not optimize for it. They approximate parts of the original network using functions of a different functional form as parts of the network. Their accounts of networks' computations are therefore not given in terms of the actual objects that are doing the computations---the network's parameters and its nonlinearities. -->

These issues motivate alternative approaches to mechanistic decomposition, including parameter decomposition methods <cite>braun2025interpretabilityparameterspaceminimizing, bushnaq2025spd, chrisman2025identifyingsparselyactivecircuits</cite>, which give accounts of network function in terms of *parameter components* that the network uses on each datapoint. *Ablation-based parameter decomposition methods* <cite>braun2025interpretabilityparameterspaceminimizing, bushnaq2025spd</cite> identify a set of parameter components where as few components as possible are "necessary" to perform the same computations original network on any datapoint, where "necessary" means that they cannot be ablated (including, crucially, partial ablations) on a given datapoint without adversely affecting output reconstruction error. Simultaneously, the parameter components are selected to implement as simple computations as possible and to sum collectively to the target network's parameters. If parameter components exhibit all these properties, then they are strong candidates for the network's "ground truth mechanisms" (though one would first need to accept philosophically that such mechanisms can be said to exist in non-toy networks!).

Parameter decomposition methods can identify known ground truth mechanisms in toy models that: are not aligned to e.g. neurons, individual attention heads, or layers; operate on representations in superposition; or are multidimensional. And, due to the requirement that components sum to the target model, parameter decomposition methods should not exhibit feature splitting. Notably, parameter decomposition methods are also architecture-agnostic and can readily be applied to any architecture, unlike activation-based methods, where it has been challenging to use the same decomposition methods to decompose both attention layers and MLPs <cite>kamath2025tracing, ameisen2025circuit, wynroe2024decomposing, ge2024localglobal</cite>. In demonstration of this ability, previous work has used ablation-based parameter decomposition to identify induction heads in a transformer trained on a toy model of induction <cite>christensen2025decomposition</cite>.

Ablation-based parameter decomposition methods thus promise solutions to many of the issues of activation-based decomposition methods. However, prior parameter decomposition proposals have several important shortcomings:

- **No application to full language models** While the most recent parameter decomposition method, Stochastic Parameter Decomposition (SPD)<cite>bushnaq2025spd</cite> is more scalable than its predecessor (Attribution-based Parameter Decomposition <cite>braun2024identifying</cite>), it has not yet been applied to full language models.
- **No demonstration of robustness to adversarial ablations**: While some work has applied SPD to a single layer of GPT2-small <cite>christensen2025decomposition</cite>, no application of SPD so far has measured key metrics that would be necessary to ensure mechanistic faithfulness, such as output reconstruction under adversarial ablations (rather than only stochastic ablations).
- **Partial incompleteness: No clustering from subcomponents to components**: Previous implementations of SPD have also been partially incomplete: Attribution-based parameter decomposition <cite>braun2025interpretabilityparameterspaceminimizing</cite> decomposed networks into full vectors in parameter space, which span all parameters in the model. But SPD decomposes them into rank-one matrices, which are limited only to single parameter matrices. A full implementation of SPD requires a *post hoc* clustering step to combine multiple rank-one matrices into full vectors in parameter space, but previous work left this clustering step implicit <cite>bushnaq2025spd</cite>.
- **No analysis of nonlinear interactions between components**: Previous work omitted analyses of the nonlinear interactions between parameter components, which would be crucial for assessing how useful parameter decomposition methods are for interpretability.

In this work, we resolve all of these issues and introduce a method called *ad**V**ersarial **P**arameter **D**ecomposition* (**VPD**).

VPD builds heavily on the SPD method introduced by <cite>bushnaq2025spd</cite> but has several important modifications, which together make it more mechanistically faithful and scalable to larger models than decomposed in previous work (cref section). The primary difference between VPD and SPD is the way ablations are done. On each datapoint, both SPD and VPD sample from the space of possible partial ablations of parameter components in order to check whether those parameter components can be partially ablated in any combination, thus identifying whether they are "necessary" for that datapoint. But SPD samples from the space of partial ablations using *stochastic* samples from the space, whereas VPD uses *adversarially chosen samples*. Both approaches are nonetheless designed to approximate what would happen if we could check *all* potential partial ablations.

Here, we use VPD to decompose a small language model ($67$M parameters) trained on the Pile <cite>gao2020pile</cite>. We find parameter components that are highly interpretable (cref section), both in terms of the dataset examples that they activate on (cref section) and how they interact with other components to produce specific behaviors (cref section). We compare the parameter components that we find to the objects found by other decomposition methods, such as per-layer and cross-layer transcoder (CLT) latents and find that they explain more of the target model's performance using an equivalent number of active components (cref section); exhibit less feature splitting (cref section); have comparable or greater interpretability (cref section); and are more mechanistically faithful (cref section). We develop attribution graphs that let us study the circuits that underlie some language model behaviors.
Furthermore, we analyze the nonlinear interactions between parameter components (cref section). We demonstrate that complex nonlinear interactions are rarer than would be expected by chance, despite not being a property our method optimizes for directly, suggesting that it reflects an underlying computational simplicity in the target model itself. Finally, we demonstrate we can use our understanding of the network's parameters to rewrite its neural algorithm for emoticon predictions (cref section).

## The core method: AdVersarial Parameter Decomposition

<label id="sec:method"/>
<!-- Dan (25th March): I think the Methods section should be 2 pages with the core formulas and minimal explanation, and the rest in the appendix. Maybe this could be different in an HTML version. But yeah I may be out of date with what's happening here. -->

Our method, VPD, builds heavily on SPD <cite>bushnaq2025spd</cite>. Our explanation of VPD does not assume familiarity with SPD <cite>bushnaq2025spd</cite> or its predecessor <cite>braun2025interpretabilityparameterspaceminimizing</cite>. In this section, we introduce ablation-based parameter decomposition methods from scratch and highlight key differences between VPD and prior methods in this class.

Our goal is to decompose a neural network into the *mechanisms* that it uses to compute its behavior—the things that it uses to take input activations, compute its hidden activations, and finally compute its output. We don't approach this goal with strong presuppositions of what a "mechanism" is. But we take for granted that a typical network doesn't use all of its mechanisms on every input (or, at least, it doesn't use all of its mechanisms by the same amount). If that were not the case, then networks could not be said to be *modular*, having distinct parts that do different things on different inputs. Without modularity, networks simply couldn't be decomposed into distinct functional units.

One candidate here for the network's mechanisms is the network's parameters. Like mechanisms, networks appear not to use all of their parameters simultaneously on every datapoint <cite>veit2016residual, zhang2022moefication, dong2023attention</cite>. This happens, for instance, when a network's parameters "read from" activation subspaces that are orthogonal to the activations on that datapoint, thus projecting the activations to zero, thereafter having no downstream causal effect. Alternatively, if the activations fail to "activate" a given ReLU neuron, the activation of that neuron is zero, thereafter having no downstream causal effect. However, the network's parameters are in fact a single vector in the network's parameter space, and do not have an obvious decomposition into parts. How should they be decomposed into parts that comprise the network's mechanisms?

On a high level, parameter decomposition methods use the idea that it should be possible, for a given datapoint, to identify the "subset" of the network's parameters that are necessary and sufficient for computing its output on that datapoint. That "subset" of parameters should contain all the mechanisms used by the network on that datapoint. If particular "subsets" of the networks parameters are repeatedly used together by different datapoints, then they may be part of the same mechanism. Parameter decomposition methods therefore aim to find particular "subsets" of the networks parameters that tend to be used together, where as few of them as possible are necessary and sufficient for computing the network's output on any input.

More concretely: If particular parameters are unused by the network on a particular datapoint, then we should be able to ablate them (including partially) on that datapoint without adversely affecting the network's output. Ablation-based parameter decomposition methods thus aim to decompose network parameters into a set of vectors in parameter space called *parameter components*. Parameter components are trained to exhibit a number of specific properties such that, if they exhibit those properties, they would be good candidates for the network's "mechanisms". They are trained to be:

- **Parameter-faithful**: They sum to the network's total parameter vector;
- **Minimal**: As few components as possible are causally important for computing the network's output on any particular input;
- **Mechanistically faithful**: Every subset of components that includes the causally important components is sufficient to compute the network's output;
- **Simple**: Components should each involve as little computational machinery as possible.

In the following sections, we define parameter components concretely and define how they are optimized to exhibit each of these four properties.

### Parameter components are vectors in parameter space and consist of subcomponents

<label id="sec:method_components"/>
Suppose we have a neural network $f(x;\theta)$ with parameters $\theta$. We would like to decompose this parameter vector into a sum of *parameter components* $\theta = \sum_i \theta_i$ with the above properties.

It would be computationally expensive to decompose models into whole parameter vectors, since each such vector would have a memory cost equivalent to the whole target model. Therefore, as in <cite>bushnaq2025spd</cite>, we use a less expensive way to parameterize parameter components. Although its parameters $\theta$ can be expressed as a single large vector, they are more commonly conceptualized as a set of matrices $\theta = \{W_1, \dots, W_L\}$. We further decompose individual matrices into sums of rank-one matrices called *subcomponents*, each parametrized as an outer products of two vectors: $W_l \approx \sum_{c} \vec{U^l_c} \vec{V_c^{l \top}} = U_l V_l^\top $, where there may be more subcomponents than rows and columns in the matrix. Although a single subcomponent explicitly parameterizes only a single weight matrix, it implicitly parametrises a full parameter vector if we assume it takes values of $0$ in every other weight matrix. It is therefore possible to combine these subcomponents into full parameter components by adding them together in the right way. We identify these components using a subcomponent clustering method. Previous work left this clustering step implicit, but in this paper we introduce an explicit method (<ref>app:clustering</ref> appendix clustering method).

<figure>
<label id="fig:sum_components"/>
<img src="figures/Sum of components.png">
<figcaption>Parameter decomposition methods decompose target model parameters into vectors in parameter space (parameter components) that are optimized to approximate the model's mechanisms. </figcaption>
</figure>

### Enforcing parameter faithfulness with a $\Delta$-component

To ensure the components collectively sum to the parameter vector of the target model, we define additional Delta-components $\Delta^l$ that parametrize the difference between our subcomponents and the original model's matrices:

$$\Delta^l_{i,j}:=W^{l}_{i,j}- \sum^C_{c=1} U^l_{i,c} V^l_{c,j}$$
<!-- LaTeX original:
\Delta^l_{i,j}:=W^{l}_{i,j}- \sum^C_{c=1} U^l_{i,c} V^l_{c,j}
-->

Additionally, we encourage Delta-components to be approximately zero with an auxiliary MSE loss between the sum of the parameter subcomponents and each target model matrix, see Appendix <ref>app:delta_l2</ref>.

<!-- Optimizing for faithfulness is straightforward. As in \cite{bushnaq2025spd} and \cite{braun2025interpretabilityparameterspaceminimizing}, we simply penalize the mean-squared error between the target model parameters and the sum of the subcomponents: -->

### Optimizing for minimality

<label id="sec:opt-minimality"/>

For minimality, we want as few components as possible to be causally important for computing the network's output on any particular input. We therefore need some way to estimate which parameter components are "required" for computing the network's output on a given datapoint. We also require a notion of how well the "required" subcomponents have reconstructed the network's output.

Ablation-based parameter decomposition methods contend that a parameter component is "required" if it cannot be ablated without affecting the model's output on that datapoint. As in <cite>bushnaq2025spd</cite>, we train a *causal importance function* $\Gamma: X \rightarrow [0,1]^{C \times L}$ to predict how ablatable each subcomponent is on a given datapoint. Like <cite>bushnaq2025spd</cite>, we implement $\Gamma$ as a neural network, though we use a different architecture (Appendix <ref>app:ci_function</ref>).

We want our causal importance function to output *causal importance values* $g^l_c(x,t)\in[0,1]$ for each subcomponent $(c)$ of weight matrix $l$ on a given datapoint $x$ at sequence position $t$. If $g^l_c(x,t) = 0$, then that subcomponent should be fully or partially ablatable without affecting the output. If $g^l_c(x,t) = 1$, then it should not be possible to ablate that component without affecting the model's output on that datapoint.

We also want our causal importance values to predict the maximal extent of the ablatability of each subcomponent. Otherwise, the causal importance function could output a value of $1$ for every subcomponent on every input. We must therefore train the causal importance values $g^l_c(x,t)$ to take minimal values.

Together, this leads to our *importance minimality loss*:

<label id="eq:minimal"/>
$$\begin{aligned}
\mathcal{L}_{\text{importance-minimality}}=\frac{1}{BT}\sum_{b,t}\sum^L_{l=1}\sum^C_{c=1} \vert g^l_c(b,t) \vert^p,
\end{aligned}$$
<!-- LaTeX original:
\label{eq:minimal}
\begin{aligned}
\mathcal{L}_{\text{importance-minimality}}=\frac{1}{XT}\sum_{x,t}\sum^L_{l=1}\sum^C_{c=1} \vert g^l_c(x,t) \vert^p,
\end{aligned}
-->


where $p>0$. <!-- todo make average over batches if that's what we do. %todo make other losses consistent too. -->

The Delta-components $\Delta^l$ are defined always to have causal importance values of zero, since they should never be required to compute the model output.

### Optimizing for mechanistic faithfulness

<label id="sec:opt-mech-faithfulness"/>
<!-- \subsubsection{Aggregating local explanations into global ones requires a careful definition of mechanistic faithfulness} -->

#### Mechanistic faithfulness implies the ability to aggregate local explanations into global ones

<label id="sec:mech-faith-aggregate"/>

Parameter decomposition methods aim to identify the simplest, fewest parameter components that are "causally important" on a datapoint. This is quite a bold aim, since actually achieving this goal means accurately describing the network's causal structure in the simplest possible terms. Implicitly, this is a claim about mechanistic faithfulness, since it would be impossible to accurately describe the network's causal structure without it. This means it is quite important to get our definition of "causal importance" right, because it bears directly on our claims of mechanistic faithfulness. Earlier, we said a parameter component is considered "causally important" on a datapoint if it cannot be ablated without affecting the model's output. But what should this mean exactly?

Thinking about an edge case will help us understand how we should define this: Suppose two components $\theta_A, \theta_B$ can be *jointly* ablated, but not *individually* ablated, on a data point $x_1$ without affecting the output. This could happen if $\theta_A$ and $\theta_B$ cancel each other out by influencing the final model output vector in opposite directions. Should these two jointly ablatable parameter components be considered ablatable on datapoint $x_1$?

<!-- Lucius version (old - delete if you are happy with the above version): -->
<!-- For the purposes of our parameter decomposition technique: No. It is not enough for the parameter components to be ablatable jointly. We also require them to be ablatable individually and in any other possible combination of subset ablations. This stricter definition ensures that the computation performed by the causally important parameter components on a data point is mechanistically faithful to the computation the original model performed on that data point. More specifically, it ensures mechanistic faithfulness in the sense that local explanations of the model's behavior on single data points or small data subsets involving subsets of parameter components can aggregate into more global explanations of the network's behavior over larger subsets of the data in the way we expect. -->

<!-- Lucius version (old): Continuing our example above: Suppose we develop an explanation of the network's behavior on a data point $x_1$, based on some subset of parameter components $S_1$, and then later develop an explanation of the network's behavior on some other data point $x_2$, using some other subset of parameter components $S_2$ which includes $\theta_1$, but does not include $\theta_2$. We want the union of both subsets $S_1 \cup S_2$ to still produce the same behavior on $x_1$ and $x_2$. Otherwise, our explanations have failed to capture how a single parameter vector can produce the observed outputs on both data points. If $S_1$ did not include both $\theta_1$ and $\theta_2$, this criterion would not be satisfied, because then $S_1 \cup S_2$ would contain $\theta_1$ but not $\theta_2$, breaking the cancellation between them. Thus, we conclude that $\theta_1$ and $\theta_2$ should both be considered causally important on $x_1$. -->

For our purposes: No. It is not enough for the parameter components to be ablatable jointly. We also require them to be ablatable individually and for all other possible combinations of ablations of other components. This stricter requirement is important for mechanistic faithfulness. Suppose the action of $\theta_A$ is to add a vector $v$ to the residual stream and the action of $\theta_B$ is to add $-v$ in a later layer. They may be jointly ablatable, but removing them would be mechanistically unfaithful to how the network actually computed its output!

This stricter requirement is also crucial for the goals of mechanistic interpretability in another sense: It ensures that *local explanations* of the model's behavior on single data points (or small subsets of the dataset) involving subsets of parameter components can aggregate into more *global explanations* of the network's behavior over larger subsets of the data in the way we expect. To understand why local-to-global aggregation of explanations only works under this stricter requirement, let's continue the example:
<!-- TODO(Lee): Resolve: 23 March, 3:53 pm, I am considering mentioning the high level claim here, but maybe putting the pseudo-proof-by-contradiction below in either the disucssion section or appendix. -->

Suppose we explain the network's behavior on a data point $x_1$ using some subset of parameter components $S_1$. And imagine, since $\theta_A$ and $\theta_B$ are jointly ablatable, that we (wrongly) decide they aren't causally important on $x_1$, and therefore both $\theta_A, \theta_B \notin S_1$.
Now suppose we later explain the network's behavior on some other data point $x_2$, using some other subset of parameter components $S_2$. But on this datapoint, it happens to be the case that $\theta_A \in S_1$ but $\theta_B \notin S_2$.
It should be the case that the network parameterized by the union $S_1 \cup S_2$ produces the same behavior on $x_1$ and $x_2$ as each of those subsets on their respective datapoint. In other words, it should be the case that

$$f(x_1, \sum_{\in S_1} \theta_i) \approx f(x_1, \sum_{\in S_1 \cup S_2} \theta_i) \quad\text{ AND }\quad f(x_2, \sum_{\in S_2} \theta_i) \approx f(x_2, \sum_{\in S_1 \cup S_2} \theta_i).$$
<!-- LaTeX original:
f(x_1, \sum_{\in S_1} \theta_i) \approx f(x_1, \sum_{\in S_1 \cup S_2} \theta_i) \quad\text{ AND }\quad f(x_2, \sum_{\in S_2} \theta_i) \approx f(x_2, \sum_{\in S_1 \cup S_2} \theta_i).
-->

This is because $S_1$ should contain all the parameter components that are causally important on $x_1$, and $S_2$ should contain all the parameter components that are causally important on $x_2$. So their union should behave similarly on their respective datapoints. But because we (wrongly) decided earlier that $\theta_A, \theta_B \notin S_1$ because they were jointly ablatable, now only $\theta_A \in S_1 \cup S_2$. Unfortunately $\theta_B \notin S_1 \cup S_2$. This means that in $S_1 \cup S_2$ there is no $\theta_B$ to cancel the effects of $\theta_A$. So

$$f(x_1, \sum_{\in S_1} \theta_i) \not\approx f(x_1, \sum_{\in S_1 \cup S_2} \theta_i).$$
<!-- LaTeX original:
f(x_1, \sum_{\in S_1} \theta_i) \not\approx f(x_1, \sum_{\in S_1 \cup S_2} \theta_i).
-->

This is an undesirable outcome that results from how we defined what it meant for $\theta_1$ and $\theta_2$ to be causally important! We therefore conclude that $\theta_1$ and $\theta_2$ should both be considered causally important on $x_1$, even though their effects cancel on that datapoint.

This example is an illustration of a more general requirement: We want the union of $S_1$ with *any* other subset of parameter components to produce approximately the same output as the target model on data point $x_1$. This way, we can be sure that our local explanation did not miss any computationally relevant behavior that would cause it to not aggregate with other local explanations in the way we expect. This property is central to our definition of "mechanistic faithfulness" and an important consequence of our definition of "causal importance".

#### Optimizing for mechanistic faithfulness: Setup

<label id="sec:opt-mech-faith-setup"/>

Ablation-based parameter decomposition methods, at their core, instantiate this definition of mechanistic faithfulness by estimating how ablatable each parameter component is (by using causal importance functions introduced in Section <ref>sec:opt-minimality</ref>) on a given sequence position of a given datapoint, then actually doing a (full or partial) ablation and training the ablated parameters to approximate the same output as the unablated model.

Formally, we define ablation masks $m^l_c(b,t,r)\in[0,1]$ for each subcomponent at each each batch index $b$ and sequence position $t$. These masks define new weight matrices $W'^l(b,t,r)$ which can take the place of the original model matrices $W^l$:

<label id="eq:masked_params"/>
$$\begin{aligned}
&W'^l_{b,t,i,j}(r):=\sum^C_{c=1} U^l_{i,c} m^l_{b,t,c}(r) V^l_{c,j} + m^l_{b,t,C+1}(r) \Delta^l_{i,j}\\
\end{aligned}$$
<!-- LaTeX original:
\label{eq:masked_params}
\begin{aligned}
&W'^l_{i,j}(x,t,r):=\sum^C_{c=1} U^l_{i,c} m^l_c(x,t,r) V^l_{c,j} \\
\end{aligned}
-->

It is important to note that the masks are not the causal importances, $g^l_{b,t,c}$. Instead, the masks are given by

$$m^l_{b,t,c}(r) :=g^l_{b,t,c}+(1-g^l_{b,t,c})r^l_{b,t,c},$$
<!-- LaTeX original:
m^l_c(x,t,r) :=g^l_c(x,t)+(1-g^l_c(x,t))r^l_c(x,t),
-->

where $r^l_{b,t,c} \in [0, 1]$. This means that if a subcomponent's causal importance is $1$, the only possible value of its mask is $1$, whereas if the causal importance is $0$, its mask can take any value between $0$ and $1$. The causal importance of the Delta-components $\Delta^l$ is always zero.
<!-- TODO(Lee): 25 March, 12:20 pm, There's room for a simple figure explaining the above 0/1 interval thing -->

It is important to understand how these position-dependent masked weight matrices are used during the forward pass. Because $W'^l(b,t,r)$ depends on the sequence position $t$, the effective weight matrix used at each position within a single forward pass is *different*. Concretely, when computing the output of layer $l$ at sequence position $t$, we replace the original weight matrix $W^l$ with $W'^l(b,t,r)$, which is constructed from the masks at that specific position. This means that during a single forward pass through the network, each layer applies a different linear transformation at each sequence position, determined by which subcomponents are masked on vs. off at that position. The masks $r^l_{b,t,c}$ are sampled independently for each $(b, t)$, so one position might have a given subcomponent active while the neighboring position does not.

In the idealised setting, we then demand that, for *all possible joint combinations* of masks $r\in {[0,1]}^{L\times B \times T \times C+1}$, the resulting masked weight matrices yield final outputs that approximately match the final outputs of the original model at every batch index and every output sequence position:

<label id="eq:subcomponents"/>
$$\begin{aligned}
%&W'^l_{b,t,i,j}(r):=\sum^C_{c=1} U^l_{i,c} m^l_{b,t,c}(r) V^l_{c,j}+m^l_{b,t,C+1}}(r) \Delta^l_{i,j} \\
&\forall r: f(x_b\vert W'^1(r),\dots,W'^L(r))\approx f(x_b\vert W^1,\dots,W^L).
\end{aligned}$$
Where $f(x_b\vert W^1,\dots,W^L)\in \mathbb{R}^{T}$
<!-- LaTeX original:
\label{eq:subcomponents}
\begin{aligned}
%&W'^l_{i,j}(x,t,r):=\sum^C_{c=1} U^l_{i,c} m^l_c(x,t,r) V^l_{c,j} \\
&\forall r: f(x\vert W'^1(x,t,r),\dots,W'^L(x,t,r))\approx f(x\vert W^1,\dots,W^L).
\end{aligned}
-->

This definition of ablatability lies at the heart of how VPD and other ablation-based parameter decomposition methods ensure that the causal importances they provide are mechanistically faithful to the original network. <!-- For more discussion, see \cite{braun2025interpretabilityparameterspaceminimizing}. -->

#### Optimizing for mechanistic faithfulness: Reconstruction under *stochastic* ablations

<label id="sec:opt-mech-faith-stoch"/>

We can use an output reconstruction loss to train the masked model's output to approximate the target model's. Unfortunately, to ensure we satisfy Equation <ref>eq:subcomponents</ref>, we would need to do this for *all possible values of* $r\in {[0,1]}^{C\times L},$ which is a high dimensional continuous interval, making such a loss impossible to compute exactly. However, a key insight of Bushnaq et al. <cite>bushnaq2025spd</cite> was that it is possible to *approximately* minimize reconstruction loss on all values in that interval using a finite number $S$ of uniform random samples $r^{l,(s)}_{c}(b,t) \sim \mathcal{U}(0,1)$ for every sequence index $t$ and every batch index $b$. These samples can be used to create stochastic masks $m^l_c(b, t, g^l_c(b, t)) \sim \mathcal{U}(g^l_c(b, t), 1)$, and minimize reconstruction loss on that finite number of samples. This leads to the *stochastic reconstruction loss*:

<label id="eq:random_recon"/>
$$\begin{aligned}
\mathcal{L}_{\text{stochastic-recon}}&=\frac{1}{S}\sum^S_{s=1}\frac{1}{BT}\sum_{b,t} D \left( f(b,t\vert W'(x,t, r^{(s)})),f(x\vert W) \right) \\
\end{aligned}$$
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

where $D$ is an appropriate divergence measure in the space of model outputs, such as KL-divergence or mean squared error. In practice, we find that using one sample ($S=1$) produces similar training behavior as using more samples. Additionally, for better convergence, we train by sampling masks for randomly chosen subsets of the model's matrices instead of all layers simultaneously (Appendix <ref>app:subset_recon</ref>).

The $\mathcal{L}_{\text{importance-minimality}}$ and $\mathcal{L}_{\text{stochastic-recon}}$ losses were introduced by <cite>bushnaq2025spd</cite> to optimize parameter components to replicate the target model's outputs while using as few parameter components as possible. While in many toy settings these losses are enough to succeed, our attempts to apply ablation-based parameter decompositions at larger scales (such as language models) revealed several pathologies that were missed by <cite>bushnaq2025spd</cite>. Prior work under-appreciated the importance of adversarial ablatability (Section <ref>sec:methods_adv</ref>) and parameter component simplicity (Section <ref>sec:methods-simplicity</ref>), which we address in the next sections.

#### Optimizing for mechanistic faithfulness: Reconstruction under *adversarial* ablations

<label id="sec:methods_adv"/>
VPD optimizes for adversarial ablatability of parameter components that are "causally unimportant" on a datapoint, which is a stricter criterion than SPD's stochastic ablatability.
SPD's $\mathcal{L}_{\text{stochastic-recon}}$ loss does, however, in the limit of infinite stochastic samples and perfect reconstruction, approximate our desired condition from Equation <ref>eq:subcomponents</ref>. But we don't have time to do infinite samples. And equation <ref>eq:subcomponents</ref> requires that the masked model approximates the target model well for *all* possible values of $r$, not just on average. Thus, if the reconstruction loss isn't exactly zero, which will essentially never happen in practice, stochastic sampling can greatly underestimate the worst-case reconstruction error for values of $r$ that are sampled adversarially to maximize reconstruction loss. We found that training without an adversarial sampling scheme produces decompositions for which adversarial sampling can find values of $r$ that have worse-than-random reconstruction loss, which is not permitted under Equation <ref>eq:subcomponents</ref> (See also Appendix Figure <ref>app:fig:adv_vs_no_adv</ref>).

VPD therefore introduces an adversarial loss to help ensure this property:

<label id="eq:adv_recon"/>
$$\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}&=\max_{r^{\text{adv}}} D \left( f(x\vert W'(x,t,r^{\text{adv}})),f(x\vert W) \right) \\
\end{aligned}$$
<!-- LaTeX original:
\label{eq:adv_recon}
\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}&=\max_{r^{\text{adv}}} D \left( f(x\vert W'(x,t,r^{\text{adv}})),f(x\vert W) \right) \\
\end{aligned}
-->

However, if the adversarial sampler were completely unconstrained, it would actually be too strict: Some decompositions that we would intuitively regard as valid would be effectively excluded by it. For example, in many theoretical toy models of circuits in superposition <cite>hänni2024mathematicalmodelscomputationsuperposition, bushnaq2024circuits, linsefors2025circuits</cite> models can contain more circuits than neurons, only some of which are used by the model on any given forward pass. However, the inactive circuits each still contribute some small interference "noise" to the computation. Since this noise is uncorrelated between circuits, its overall size remains small enough that the interference doesn't "break" the computation. We would like to consider these inactive circuits not to be causally important, since the model is in some sense not really using them to compute the output. But if we chose the absolute worst-case $r^{\text{adv}}(x)$ on every data point in such a model (which we can do if we have a completely unconstrained adversarial sampler), we could, for example, ablate all inactive circuits which contribute noise with a negative sign, but keep all inactive circuits which contribute noise with a positive sign. This would vastly increase the overall size of the noise and thus change the final output of the model!

In general, we want the adversarial sampler to penalise *systematic* defects in the decomposition, but we do not want it to exploit random noise by overly fine-tuning its choice of ablations to particular data points. In practice, when using the decomposition to understand or edit the target model, we usually care about the behavior of particular component maskings over multiple data points, rather than the behavior of all possible maskings on single data points. <!-- TODO(Lucius): (Lee: I think we need to be a bit clearer about what we mean by 'systematic') -->
<!-- TODO(Lucius): (Lee: This sentence is a bit unclear to me. @lucius could you reword please?) -->

Thus, in order to force the adversarial sampler to rely on systematic flaws in the decomposition instead of fine-tuning to every data point, we restrict it to use the same $r^{\text{adv}}$ on all elements in a batch. Ideally, we would like to use the same sources for the whole data set, but this would be much more computationally expensive. In practice, we actually only use this shared source scheme for evaluation. In training, we further save on cost by adversarially sampling $r^{\text{adv}}(x)$ independently for different elements in a batch, but keeping the same persistent $r^{\text{adv}}(x)$ across batches. For more details on how adversarial sampling is performed in practice, see Appendix <ref>app:methods</ref>.

### Optimising for simplicity

<label id="sec:methods-simplicity"/>

The "simplicity" of parameter components is supposed to capture the notion that each parameter component uses as little computational machinery as possible. Otherwise, we could say that the target model is one big parameter component, and proclaim our decomposition as complete without doing any actual decomposition!

One of the reported benefits of SPD <cite>bushnaq2025spd</cite> over APD <cite>braun2025interpretabilityparameterspaceminimizing</cite> was that SPD used rank-one subcomponents, and where as few of these rank-one subcomponents are necessary to reconstruct the output. In Bushnaq et al. <cite>bushnaq2025spd</cite> we believed this meant that SPD did not need dedicated losses to optimize for "simplicity" even though <cite>braun2025interpretabilityparameterspaceminimizing</cite> did.

Our optimism in Bushnaq et al. <cite>bushnaq2025spd</cite> was misplaced. Unfortunately, some rank-one solutions are "simpler" than others (Figure <ref>fig:simplicity</ref>). It is possible to add multiple rank-one mechanisms together and for their sum also to be rank-one as long as either their right or left singular vectors are equal <footnote>For a theoretically clean motivating example of this phenomenon, see the toy model of ping pong superposition <cite>gibson2025</cite>.</footnote>.
<!-- \footnote{A theoretically clean motivating example of such a is the toy model of ping pong superposition \cite{gibson2025}. In the ping pong superposition construction, $64$ superposed rank $1$ circuits can be implemented in layers of width $21$. Only one circuit is ever active at a time, and groups of eight circuits each share the exact same origin or target neurons. Components for circuits in the same group can then be summed, and the result will again be exactly representable as a rank $1$ matrix, which is causally important for computing the output exactly when any of the circuits in the group are causally important for computing the output. Hence if we apply VPD to this toy model, the importance minimality loss alone will provide no incentive to further separate the eight rank $1$ matrices for the eight circuit groups into $64$ rank $1$ matrices for the $64$ individual circuits, leaving us with components that activate polysemantically and contain more computational machinery than they need to.} -->

We observed indications that some SPD decompositions suffered from this failure mode: Sometimes given subcomponent seemed to be involved in multiple (usually two) unrelated computations, which depended on whether the activations had strong positive or negative inner products with the subcomponent's right singular vector.

<figure>
<label id="fig:simplicity"/>
<img src="figures/simplicity.png">
<figcaption>Without the simplicity penalty, some parameter components can contain more 'computational machinery' than they should</figcaption>
</figure>

We therefore need a "simplicity" loss to incentivize parameter components to be involved in as few separable computations as possible (beyond the extent to which the importance minimality loss and rank-one constraint already encourage aspects of "simplicity" as outlined in <cite>bushnaq2025spd</cite>). To achieve this, we use the following loss:

<label id="eq:freq_minimality"/>
$$\begin{aligned}
\mathcal{L}_{\text{frequency-minimality}}=\sum^L_{l=1}\sum^C_{c=1}\sum_{x,t} \vert g^l_c(x,t) \vert^p\log_2(1+\sum_{x',t'} \vert g^l_c(x',t') \vert^p),
\end{aligned}$$
<!-- LaTeX original:
\label{eq:freq_minimality}
\begin{aligned}
\mathcal{L}_{\text{frequency-minimality}}=\sum^L_{l=1}\sum^C_{c=1}\sum_{x,t} \vert g^l_c(x,t) \vert^p\log_2(1+\sum_{x',t'} \vert g^l_c(x',t') \vert^p),
\end{aligned}
-->

This loss optimizes components to activate infrequently, since it penalizes subcomponents for being causally important (the $\sum^L_{l=1}\sum^C_{c=1} \vert g^l_c(x,t)\vert^p$) term on the left) but penalizes the subcomponents that activate more often *more* (the $\log_2(1+\sum_{x',t'} \vert g^l_c(x',t') \vert^p)$ term on the right, which sums over datapoints $x',t'$ in a batch).
<!-- TODO(Lucius): (Lee: I feel like the transition the above sentence and the below one is a bit of a nonsequitur -->
In information theory terms, the $\log_2(1+\sum_{x',t'} \vert g^l_c(x',t') \vert^p)$ term can be thought of as quantifying the bits of precision needed to specify a component well enough to obtain low loss<footnote>The more often a component is causally important, the more precisely we need to define it to obtain low total reconstruction over $X$ data points. If it is always causally important, every bit of precision in the component's definition we get wrong can hurt our loss on every data point. If it is rarely causally important, every bit we get wrong will only hurt our reconstruction loss on some fraction $\frac{\sum_{x,t} \vert g^l_c(x,t)\vert^0}{XT}$ of the data. Thus, we can afford to store it to fewer bits of precision without increasing reconstruction KL divergence too much. Under some simplifying modeling assumptions, this trade-off scales with $\log_2(\frac{\sum_{x,t} \vert g^l_c(x,t)\vert^0}{XT})$. We approximate the $L^0$ norm with an $L^P$ norm, and add $1.0$ to the argument for stability, since $p$-norms with $0<p$ can otherwise take values below $1.0$ and turn the $\log_2$ term negative. See Appendix <ref>app:frequency_penalty</ref> for details of the derivation of this loss.</footnote>.

There are probably multiple ways to optimize for the computational simplicity of parameter components, and we are not confident this choice is optimal (nor for the other losses). Nonetheless, we remark that the new loss introduces an interesting symmetry with the importance minimality loss: 

- $\mathcal{L}_{\text{importance-minimality}}$ encourages *datapoints* in the training set to activate as few *subcomponents* as possible; minimizing it incentives decompositions to employ *fewer* components.
- $\mathcal{L}_{\text{frequency-minimality}}$ encourages *subcomponents* to activate on as few *datapoints* in the training set as possible; minimizing it incentives decompositions to employ *more* components.

We note that this tradeoff avoids creating the same problem as "feature splitting" due to the tradeoffs between the output reconstruction, importance minimality, and frequency minimality losses (Section cref xxfeature-splitting-results-todo).
<!-- \textit{(TxDx potentially: figure illustrating the effects of the two losses on causal importances)(TxDx experiment: showing CIs on real data, where y axis is C and data index is on X axis, where the C activations are they're hierarchically clustered - should show that each component activates on fewer parameter components and there are fewer 'bias-like' components).} -->

### Summary of loss terms

<label id="sec:methods-summary"/>
In total, our loss function has five terms:

$$\begin{aligned}
\mathcal{L}_{\text{VPD}}=\mathcal{L}_{\text{adversarial-recon}}+\mathcal{L}_{\text{stochastic-recon}}+\mathcal{L}_{\text{importance-minimality}}+\mathcal{L}_{\text{frequency-penalty}}+\mathcal{L}_{\text{Delta-L2}}
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
\mathcal{L}_{\text{VPD}}=\mathcal{L}_{\text{adversarial-recon}}+\mathcal{L}_{\text{stochastic-recon}}+\mathcal{L}_{\text{importance-minimality}}+\mathcal{L}_{\text{frequency-penalty}}+\mathcal{L}_{\text{Delta-L2}}
\end{aligned}
-->

- The $\mathcal{L}_{\text{adversarial-recon}}$ and $\mathcal{L}_{\text{stochastic-recon}}$ losses optimise for **mechanistic faithfulness**.
- The $\mathcal{L}_{\text{importance-minimality}}$ loss optimises for **minimality**.
- The $\mathcal{L}_{\text{frequency-penalty}}$ loss optimises subcomponents for **simplicity**. They are also constrained to be rank-1 matrices, which is another form of simplicity.
- The $\mathcal{L}_{\text{Delta-L2}}$ auxilliary loss optimises subcomponents to be **parameter-faithful**, even without the $\Delta$-components, which ensure it.

The $\mathcal{L}_{\text{adversarial-recon}}$ and $\mathcal{L}_{\text{frequency-minimality}}$ losses represent the key differences in our approach compared with <cite>bushnaq2025spd</cite>. However, there are several other, smaller differences that do not fundamentally change the method but that we found helpful for decomposing language models. For more details of our method, see Appendix <ref>app:methods</ref>.

We evaluate the quality of our decomposition on a number of key metrics. For assessing the quality of a decomposition, the most important are $\mathcal{L}_{\text{adversarial-recon}}$ and $L_0$ per datapoint. For readers looking for practicable advice on what hyperparameter to tune and what metrics to care for, we have provided a detailed "Training recipe for VPD" in the appendix (Appendix <ref>app:sec:recipe</ref>).

## Decomposing a language model into parameter components using VPD

### The language model that we decomposed

<label id="sec:langauge-model-details"/>

<!-- We trained a four-layer, 67M parameter decoder-only transformer model on an uncopyrighted subset of The Pile \cite{gao2020pile}. It uses standard multihead attention layers \cite{vaswani2017attention} with RoPE positional encoding and MLPs with a GELU activation function. RMSNorm is applied to the inputs of the attention and MLP layers \cite{xiong2020layernormalization} and before the final unembedding layer. The token embedding and LM head weights are tied, giving the model approximately 28M non-embedding parameters and 67M total parameters. The model achieves a final validation cross-entropy loss of approximately 2.71. The model architecture is summarized in Table \ref{model-hyperparams}; full training details can be found in Appendix \ref{app:training-details}. -->
We trained a four-layer 67M parameter decoder-only transformer model on an uncopyrighted subset of The Pile <cite>gao2020pile</cite>. A summary of the model architecture and training results can be found in Table <ref>model-hyperparams</ref> and full training details of our target model can be found in Appendix <ref>app:training-details</ref>.

<label id="model-hyperparams"/>
| **Property** | **Value** |
|---|---|
| Layers | 4 |
| Residual stream $d_{\text{model}}$ | 768 |
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
| Final validation cross-entropy loss | 2.71 |
*Architecture and other attributes of our target language model.*

<!-- Although most of the results in this paper focus on our decomposition of this language model, we show that VPD achieves canonical decompositions of the toy models decomposed in \cite{braun2025interpretabilityparameterspaceminimizing} and \cite{bushnaq2025spd} in the appendix. -->

It is worth noting that, even though transformer models share parameters at each sequence index, they usually perform different computations at each sequence index because different sequence indices can have different activations. Our causal importance functions therefore output different causal importance values for each sequence position, thus activating different sets of parameter components at different points in the sequence.

<figure>
<label id="fig:placeholder"/>
<img src="figures/archi.png">
<figcaption>Our target model is a standard 4-layer transformer language model.</figcaption>
</figure>

### Parameter components approximate the target model relatively well

<label id="sec:approx-target-well"/>
  
<!-- TODO once Bart is done editing this section we'll move some of this to appendix to keep the narrative tight -->
<!-- Some summary statistics of the trained decomposition -->

<!-- TODO(Bart): More details about transcoder/CLT training -->
<!-- TODO(Bart): Appendix table with all the properties (e.g alive components etc) -->

If a decomposition method has correctly identified the mechanisms underlying a model's computation, then activating only the mechanisms that the method identifies as important on a given input should approximately reproduce the model's behavior on that input. Conversely, if a replacement model fails to reproduce the model's behavior, then the decomposition has either missed important mechanisms or identified spurious ones. Reconstruction quality is therefore a necessary (though not sufficient) condition for a decomposition to be mechanistically faithful.

In this section, we compare VPD's reconstruction quality against two families of activation-based decomposition methods: per-layer transcoders (PLTs) <cite>dunefsky2024transcodersinterpretablellmfeature</cite> and cross-layer transcoders (CLTs) <cite>lindsey2024crosscoders</cite>.

**Experimental setup** 

All methods are evaluated by replacing the MLP layers of our target model with their sparse reconstructions and measuring the resulting increase in cross-entropy loss relative to the unmodified target model. We simultaneously replace all 4 MLP layers unless otherwise noted. For VPD, sparsity is determined by a causal importance threshold: subcomponents with causal importance below the threshold are ablated. We report results at thresholds $0$ (retaining all subcomponents with any nonzero causal importance) and $0.5$. For PLTs and CLTs, sparsity is controlled by the number of active features $k$ per module, with BatchTopK training at $k \in \{8, 16, 32, 64\}$. We train both PLTs and CLTs at a dictionary size of $4,096$ features per layer (similar to the number of VPD components per module) and at a dictionary size of $32,768$ features per layer, to test whether the gap can be closed by simply scaling up the activation-based methods. Each PLT/CLT is trained on 500M tokens.

Comparing sparsity across methods requires care, because each method has structurally different notion of what constitutes a single active component. A CLT feature writes to the residual stream at every layer simultaneously, while a PLT feature affects only one layer. VPD subcomponents are scoped to individual weight matrices, and each MLP layer has two such matrices (up-projection and down-projection). To ensure our conclusions are not artifacts of how we count components, we show results under three formulations of sparsity: (1) average active components per module (active encoder features for PLTs/CLTs; active components per weight matrix for VPD), (2) active components per MLP output reconstruction (adjusting for the fact that a CLT feature affects all layers and that VPD uses two modules per MLP), and (3) total active parameters (VPD's rank-one subcomponents have more parameters than a PLT feature and a single CLT feature has multiple decoder vectors).


**VPD achieves a better sparsity–accuracy tradeoff**

<label id="sec:vpd-sparsity-acc-tradeoff"/>

We first compare VPD against PLTs and CLTs trained with MSE reconstruction loss on intermediate activations, which is the standard training objective for these methods. Figure <ref>fig:pareto-mse</ref> shows CE degradation as a function of sparsity under all three normalizations. VPD achieves lower CE degradation than both PLTs and CLTs at comparable sparsity levels, and the ordering is consistent across all three normalizations. Scaling the dictionary size from 4k to 32k improves the activation-based methods but does not close the gap: even the 32k CLT at $k=64$ ($\delta \approx 0.34$) only matches VPD at CI>0 ($\delta \approx 0.32$), and VPD achieves this with far fewer active components. The 4k PLTs and CLTs perform substantially worse, with CE degradation $2$--$6\times$ higher than VPD at matched sparsity.

This advantage may stem from a difference in training signal: VPD is trained end-to-end on the model's output distribution, while the MSE-trained PLTs and CLTs optimize a local, layer-wise objective. To control for this, we next train all activation-based methods end-to-end with KL divergence on the output logits, matching VPD's training objective.

<figure>
<label id="fig:pareto-mse"/>
<img src="figures/pareto_ce_reconstruction_new.png">
<figcaption>CE degradation when simultaneously replacing all 4 MLP layers with sparse reconstructions from each method. (a) Active components per module (raw L0). (b) Active components per MLP reconstruction, adjusting for CLT's cross-layer writes and VPD's paired modules. (c) Total active parameters. VPD (purple markers) Pareto-dominates the activation-based methods under all three sparsity measures. The dashed line indicates zero-ablation (all MLP outputs set to zero). Lower is better.</figcaption>
</figure>

**End-to-end transcoders overfit to their training mode**

<label id="sec:mode-mismatch"/>

When we replace all MLP layers simultaneously, there is an important design choice: should each layer's encoder see the *clean* residual stream (as computed by the original model) or the *modified* residual stream (which includes reconstruction errors from earlier layers)? We call these the ***"clean-input"*** and ***"error-propagating"*** evaluation protocols, respectively. A third option, ***"single-layer"***, replaces only one MLP at a time, with all other layers left unmodified. For a perfectly faithful reconstruction — one that exactly replicates each MLP's computation — these three protocols would produce similar results.

We train separate sweeps of BatchTopK PLTs and CLTs ($k \in \{8, 16, 32\}$) in error-propagating and clean-input mode, as well as single-layer-trained PLTs. All use KL divergence on the output logits as the training loss, matching VPD. Each model is then evaluated under all three protocols.  Figure <ref>fig:pareto-e2e</ref> shows the results.

The activation-based methods exhibit severe brittleness to evaluation mode mismatch. In the matched setting, error-propagating-trained PLTs achieve CE degradation as low as $\delta = 0.32$, and clean-input-trained PLTs reach $\delta = 0.18$ at $k=32$ (Figure <ref>fig:pareto-e2e</ref>b). But when evaluated in the *mismatched* setting, these same models degrade catastrophically: clean-input-trained models evaluated in error-propagating mode suffer $\delta \approx 2.9$--$3.5$, roughly an order of magnitude worse. The pattern is symmetric: error-propagating-trained models fail in clean-input evaluation ($\delta \approx 1.6$--$2.2$). CLTs exhibit the same pattern. The gap between matched and mismatched performance is a factor of $3$--$20\times$.

This brittleness reveals that e.g. a PLT trained in cascading mode does not simply learn to approximate each MLP's input-output function. Instead, it learns a replacement model that *jointly* accounts for both the MLP's true computation and the systematic reconstruction errors introduced by the PLTs in earlier layers. This is a compensatory strategy rather than a faithful approximation of the original target model.

Single-layer-trained PLTs, which each see only the clean residual stream for their own layer, are the most robust of the activation-based methods, and perform best in the single-layer replacement setting ($\delta \approx 0.13$--$0.19$). However, when all four single-layer-trained PLTs are inserted simultaneously, they still exhibit meaningful degradation ($\delta \approx 0.56$--$0.99$), because each was trained in isolation and cannot account for reconstruction errors accumulating from other layers.

VPD's CE degradation, by contrast, is relatively consistent across all three evaluation protocols. At CI$>$0, VPD achieves $\delta \approx 0.32$–$0.42$ regardless of whether it is evaluated in error-propagating, clean-input, or single-layer mode. This arises because VPD's stochastic and adversarial masking during training already exposes the decomposition to a rich diversity of partial ablation patterns: on each training step, a random subset of subcomponents across random subsets of weight matrices are partially masked, which naturally covers patterns resembling both error-propagating and clean-input replacement as special cases. More fundamentally, VPD's subcomponents sum to the original weight matrices, and the masked forward pass uses the same architecture and nonlinearities as the target model. A VPD reconstruction is therefore not a different function approximating the MLP, but rather a subset of the MLP's computations.

VPD does not achieve the lowest CE degradation in each individual evaluation setting: in matched-mode evaluation, the best activation-based models outperform VPD (e.g., clean-input-trained PLTs at $k=16$ achieve $\delta \approx 0.23$ vs. VPD's $\delta \approx 0.42$ in clean-input evaluation). But we view this as the expected cost of a more faithful decomposition. A model that has been specifically optimized to compensate for a particular pattern of errors will naturally outperform one that has not learned to compensate for that specific error pattern.


<figure>
<label id="fig:pareto-e2e"/>
<img src="figures/pareto_e2e_new.png">
<figcaption>CE degradation vs. L0 (active features per module) for end-to-end KL-trained methods under three evaluation protocols. (a) Error-propagating: each encoder sees the modified residual stream. (b) Clean-input: each encoder sees the clean residual stream. (c) Single-layer replacement, averaged over layers. PLTs (blue) and CLTs (orange) perform well in their training mode but degrade by 5--20x in the mismatched mode. VPD (purple markers) is relatively stable across all three protocols. Linestyle indicates training mode: solid = error-propagating, dashed = clean-input, dotted = single-layer.</figcaption>
</figure>

### Parameter components are highly interpretable

<label id="sec:param-comps-interpretable"/>
<figure>
<img src="figures/component-coherence-temp-ugly.png" />
<figcaption>Intruder-detection scores for various CLT and PLT latents, and VPD components at different CI thresholds</figcaption>
</figure>
<!-- interp of some individual components -->
TODO(oli) Add the component explorer here

### The decomposition model behaves similarly to the target model

<label id="sec:decomp-model-behav-sim"/>

<!-- Some generations from the spd model vs the dataset. -->

TODO when we are working in HTML

### Comparisons to other decomposition methods

<label id="sec:vpd-comparison-to-other-methods"/>

### Decompositions are consistent across seeds

<label id="sec:seed-stability"/>

<!-- Geometric consistency across seeds -->

(Todo Lee)

## Interpreting parameter component circuits

<label id="sec:circuits"/>

<!-- "Why is this an attribution graph and not a computational graph" - maybe for discussion -->
<!-- "This is an attribution graph, this is what the lines in it mean" "we're only showing the direct connections. Whenever the gradients hit a node that is not a target node, we stop it there" -->

### (Attribution graph methods section)

#### Attribution graphs

<label id="sec:attr-graph-intro"/>

Intro todo post-restructring

#### Attribution calculations

<label id="sec:attr-calcs"/>

We use leverage gradients to calculate attributions between two components. In particular, we calculate the gradients between each "subcomponent activation", $a_c^l = V_c^{l \top} x^l$. However, we do not always simply use the partial derivative of the target subcomponent activation with respect to the source subcomponent activation, $\frac{\partial a_{t}}{\partial a_{s}}$. The partial derivative measures the influence of $a_{s}$ on $a_{t}$ through both direct and indirect pathways. And in models with residual streams, a component's direct effects are not limited only to those in the immediate next layer. The direct effects may skip many layers! Understanding the direct effects of a component give us the clearest mechanistic picture of its role in the network's neural algorithm. We therefore need an attribution method that can distinguish between direct and indirect effects.

Instead of using the partial derivative, we use the fact that we can control how gradients flow on the backwards pass. We take the derivative $\frac{\partial a_{t}}{\partial a_{s}}$, but we stop the gradients flowing through all components that are not the source component (Figure <ref>fig:attr-graph-expl</ref>). This avoids measuring their effects on the target node, including the indirect effects of the source node that flow through them.

<figure>
<label id="fig:attr-graph-expl"/>
<img src="figures/Explaining attribution graphs.png">
<figcaption>Caption todo</figcaption>
</figure>

This derivative approximates how sensitive the target node is to the source node. Our attribution multiplies this "sensitivity" by the strength of the activation of the source node in order to measure its overall influence. Additionally, do not want to include causally unimportant nodes in our attributions, and therefore multiply the resulting term by the causal mask value of the source subcomponent:

<!-- $\text{attr}(s \to t) = \sum_{\text{batch}} \sum_{\text{pos}} \frac{\partial a_t}{\partial a_s} \cdot a_s \cdot \text{CI}(s, \text{pos})$ -->

$$\text{attr}(s \to t) = \left( \frac{\partial a_t}{\partial a_s} \right)^* \cdot a_s \cdot g_s$$
<!-- LaTeX original:
\text{attr}(s \to t) = \left( \frac{\partial a_t}{\partial a_s} \right)^* \cdot a_s \cdot g_s
-->

where the $*$ around the partial derivative denotes stopped gradients on non-source components.

#### Graph post-processing

<label id="sec:attr-graph-post-proc"/>


VPD base training yields a set of components which sum to the target model weights, and a causal importance function that tries to predict which components can be ablated without changing the final output of the model on any particular data point. To understand the target model's behavior, we also make use of additional tools: We further prune the number of components on particular prompts down to only those involved in some particular behavior we are interested in by optimizing new causal importances to reconstruct only some aspects of the target model's output. We also compute gradient attributions between components to obtain interaction graphs that visualize how components interact with each other on the forward pass.

**Post-hoc causal importance optimization**

We can further reduce the number of components we need to analyse by keeping only those components involved in computing some particular aspect of the model output we are interested in. For example, on the prompt `The` ` princess` ` lost` ` her` ` crown` `.`, we will analyse how the model successfully predicts ` her`. We are thus only interested in components that were involved in computing this specific predicition at this specific sequence position. So we can optimise new causal importances using a cross-entropy reconstruction loss against the label ` her` on the sequence position for ` lost`, instead of a KL-divergence to all the target model's output probabilities on all sequence positions of the prompt. This allows us to further reduce the number of components we need to analyse to understand some behavior of the target model. As in VPD base training, causal importances are optimised under stochastic and adversarial sampling for the masks. For details, see Appendix section <ref>app:posthoc_ci</ref>.

**Gradient attributions**

We compute gradient attributions todo cite between pairs of causally important components in adjacent layers. Many works have pointed out issues that can cause gradient attributions to be unfaithful, such as saturated softmax functions in attention layers. We use them here merely as a supplementary tool to identify some qualitative relationships between components. For more details on the gradient attributions, see Appendix section <ref>app:gradient_attributions</ref>.

### Case studies: Interpretability in language model parameter space

<label id="sec:case-studies"/>
<!-- Circuits: Biology of a small LM examples -->

#### Case study 1: Gender for possessive pronoun


<label id="graph:princess"/>
```graph
id: princess-full
data: data/princess-full.json
details: data/princess-full-details.json
caption: Attribution graph for predicting " her" on the prompt "The princess lost her crown.", pruned with adversarial sampling.<footnote>coefficient $0.5$ for cross-entropy reconstruction with stochastic sampling, coefficient $0.5$ for cross entropy with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.09$, $p=0.3$, $2000$ optimisation steps.</footnote> There are 150 active components in total. The probability on " her" is $1.000$ with CI masking, $0.999$ with stochastic masking and $0.443$ with adversarial masking under $4$ PGD optimisation steps with step size $1$. The target model assigned $0.586$ probability to " her", indicating that this graph still isn't quite capturing all the relevant computation going on. Though without adversarial the sampling we would be blind to this fact. Compare to Figure <ref>graph:bracket_ci</ref> for a graph on the same prompt pruned without adversarial sampling.
```


<label id="graph:princess_ci_masked"/>
```graph
id: princess-minimal
data: data/princess-minimal.json
details: data/princess-minimal-details.json
caption: Attribution graph for predicting " her" on the prompt "The princess lost her crown.", pruned with causal importance masking.<footnote>coefficient $1.0$ for cross-entropy reconstruction with causal importance masking, importance minimality coefficient $1.0$, $p=0.3$, $2000$ optimization steps.</footnote> There are 6 active components in total. The probability on " her" is $0.895$ with ci masking, $0.969$ with stochastic masking and $<0.0005$ with adversarial masking under $4$ PGD optimisation steps with step size $1$. Without adversarial sampling, we might falsely think that these 6 components comprise all the relevant computation in the target model. The set of components agrees well with the core causal story suggested by our incomplete interpretation of the base graph. But it is clear that if we simply ask for a set of components that are sufficient and necessary to produce the output " her", we are missing most of the relevant computation going on in the model to determine this output. Answering correctly doesn't just require a circuit that computes ` her`, but also the suppression of other circuits in the model that predict different outputs. If we didn't use adversarial sampling to prune the graphs, we might have been blind to this fact. Stochastic sampling does slightly better, but isn't sufficient either. A graph pruned with stochastic sampling can contain only $14$ components and still assign $0.549$ probability to " her" under stochastic masking and $0.984$ with ci masking, compared to the $0.586$ assigned by the target model. Under adversarial masking, the probability is $<0.0005$.
```

On the prompt `The` ` princess` ` lost` ` her` ` crown` `.` the target model correctly predicts that ` her` follows ` lost`, assigning probability $0.586$. This requires recognizing that a possessive pronoun is likely coming next, remembering that the previous token was ` princess`, and knowing that princesses usually use female pronouns. How does the model perform this task? Can we use the SPD decomposition to follow the flow of information and see what information is processed where?

Figure <ref>graph:princess</ref> shows the attribution graph for this prompt after adversarial pruning, keeping only the components that matter for predicting ` her`. The graph has a total of $150$ active components. Working backward from the output, we see that the top two positive attributions to the output are from

1. A layer 3 attention output matrix subcomponent labeled "contexts related to women, family, and maternal health". Ablating it causes the model to predict ` his` as the top continuation instead of ` her`. In turn, this component receives the most attribution from a key component in attention layer 3 active on ` princess` which is causally important on almost every token, and a value component, likewise active on ` princess`, labeled "female pronouns and nouns". This value component, in turn, has its top attribution to a layer 0 down projection component labeled "various word roots and stems" which appears to be polysemantic. It is active on various female names and other words and sentences associated with or about women, but also on in a range of other contexts, perhaps particularly scientific ones. Its top attribution comes from an mlp layer 0 input component labeled "feminine pronouns and proper nouns". While it is indeed primarily active on feminine pronouns and nouns, there is some noticeable amount of polysemanticity to this component as well, though to a lesser extent. Again, many of the subcomponents' non-female related activations seem to come from scientific contexts. This subcomponent then connects straight to the princess embedding. In summary, this computational pathway seems to carry femaleness information from ` princess` over to ` lost`. Since the relevant key (and query) components for the first almost always fire, this seems to be done by default, rather than due to any particular conditional computation.
2. A layer 2 mlp down projection matrix component labeled "fires on prepositions and verbs predicting determiners", which seems to be causally important when the model is about to predict and object pronoun, or a determiner like ' the', ' a', or ' this'. The strongest attribution to this component by far comes from an mlp input component simply labeled "verbs", which appears to be causally important primarily on tokens that are verbs. Notably, this classification appears to be based on more than the raw token itself. For example, in the sentence `I'd` `like` `to` `do` `something` `like` `this`, the component has high activation ($12.7$, $19.1$) and causal importance $1.0$ on `do` and the first `like`, but low activation ($2.9$) and causal importance $0$ on the second `like`. This component in turn receives attribution from a diverse set of layer 0 mlp components, which connect directly to the ` lost` embedding in the input.

This would seem to suggest two core mechanisms: One which moves the femaleness attribute of ` princess` over to the next token in attention layer 3, and another which suggests that a possessive pronoun might follow the verb ` lost`.

Indeed, if we optimise for high probability on ` her` under causal importance masking only, neglecting adversarial robustness, we recover a graph of just six components, see figure <ref>graph:princess_ci_masked</ref>, which corresponds almost exactly to the most attributed components in these two pathways. The femaleness components in attention 3 output and value as well as MLP 0 down and up, leading to the ` princess` embedding for one pathway, "object pronouns and fixed phrases after prepositions" connecting to "fires on verbs" connecting to the ` lost` embedding for the other. This smaller graph even generalises somewhat to other prompts: On the input `The` ` lady` ` lost` ` her` ` crown` `.`, these same six components on the same sequence positions also predict ` her` with output probability $0.895$ under causal importance masking, and $0.275$ under stochastic masking. However, they do not at all produce the correct output under adversarial masking.

Ultimately, all the components in figure <ref>graph:princess</ref> likely play some important role in the computation, otherwise the optimization process would have pruned them from the graph. This suggests that while these six components suffice to put high output probability on ` her`, they fail to suppress the outputs of other computational pathways in the model that would predict different outputs.

We do not aspire to understand the graph in Figure <ref>graph:princess</ref> in full. For example, the top negative attribution to the output ` her` comes from an mlp layer 3 down projection component labeled "syntactic punctuation and high pmi pronouns". Looking at other activating examples for this component, it seems to be causally important whenever the model is predicting that a pronoun is coming up, but its role in that computation seems to be variable: It occurs both with positive activation, in which case it increases the probability the model assigns to predicting a pronoun, and with negative activation (such as on this prompt) in which case ablating it out of the target model decreases the probability the model assigns to pronoun predictions.


```graph
id: prince-full
data: data/prince-full.json
details: data/prince-full-details.json
caption: Attribution graph for predicting ` his` on the prompt `The` ` prince` ` lost` ` his` ` crown` `.`, pruned with adversarial sampling.<footnote>Coefficient $0.5$ for cross-entropy reconstruction with stochastic sampling, coefficient $0.5$ for cross entropy with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.05$, $p=0.3$, $2000$ optimisation steps.</footnote> There are 160 active components in total. The probability on ` his` is $1.000$ with ci masking, $0.998$ with stochastic masking and $0.383$ with adversarial masking under $4$ PGD optimisation steps with step size $1$. The target model assigned $0.512$ probability to ` his`, indicating that this graph still isn't quite capturing all the relevant computation going on. Comparing to Figure <ref>graph:princess</ref> for the ` princess`, some nodes, e.g. "third-person pronouns" in the layer 3 down projection matrix and "object pronouns and fixed phrases after prepositions" in the mlp layer 2 down projection matrix appear in both graphs and have similar attributions to the output pronoun (` his`/` her`).
```

<label id="graph:prince"/>

```graph
id: prince-minimal
data: data/prince-minimal.json
details: data/prince-minimal-details.json
caption: Attribution graph for predicting ` his` on the prompt `The` ` prince` ` lost` ` his` ` crown` `.`, pruned down to six components. The two components on ` lost`, in mlp layer 2 are the same ones as in the graph for ` princess`, see Figure <ref>graph:princess</ref>. The other four nodes occur in the adversarially masked princess graph as well, but not in the princess graph pruned with causal importance masking. They activate on both male and female dataset examples, though male activations are more common, suggesting that they form general "person pathway", with people assumed male by default. Consistent with this hypothesis, using the six components in this graph for a forward pass on "the princess lost" predicts ` his` as the top logit.
```

We find similar results on the prompt `The` ` prince` ` lost` ` his` ` crown` `.`. Figure <ref>graph:prince</ref> shows an attribution graph for this prompt after adversarial pruning, keeping only the components that matter for predicting ` his`. The graph has a total of $160$ components, similar to the one for the female case. Again, a subset of $6$ components in this graph proves sufficient to compute the correct answer. This graph is very similar in character to the $6$ component graph from the princess prompt, with one computational pathway on ` lost` which uses the same two components as the one in the princess case, and another computational pathway of four components routing from ` prince` through the layer $0$ mlp and then the layer $3$ attention. However, this pathway consists of different components than the ones in the princess example. They seem to fire in both male and female related contexts, though more often male ones. We speculate that this suggests a mechanism under which male pronoun prediction is the default unless actively contradicted. Further reinforcing this hypothesis, if we run a forward pass on the princess prompt using the six components from the prince prompt, the model predicts ` his` rather than ` her`.

We stress again that the above is far from a complete account of the meaningful computation going on in the model for these input prompts. We have merely traced out the flow of information between a subset of components that are sufficient for computing the output, which is much smaller than the subset of components that are actually involved in computing the output.

#### Case study 2: Parameter components can identify attention behaviors that are distributed across attention heads

<label id="sec:attn-prev-tok"/>


In the previous example, we have seen that the parameter components found by VPD can be used to explain how attention computes the network's behavior. Notably, we did not make reference to individual attention heads in our analysis, even though attention heads are perhaps of the primary units of analysis that have been used in previous work to study attention behaviors in transformers <cite>vig2019multiscalevis, clark2019doesbertlookat, elhage2021mathematical, olsson2022incontextlearninginductionheads, wang2022interpretability, janiak2023polysemantic, nam2025causalheadgatingframework</cite>. Unfortunately for interpretability, it is possible for attention blocks to implement behaviors using computations that are distributed across multiple heads <cite>jermyn2023attention, jermyn2025attention</cite>.

<!-- Good text: And even in cases where behaviors seem mostly localized to individual heads, these heads were not the only heads involved in that behavior, such as "previous token behavior" in "previous token heads" \cite{elhage2021mathematical, olsson2022incontextlearninginductionheads, wang2022interpretability}. Complicating matters even further, individual heads might perform multiple roles, in addition to roles being distributed across multiple heads \cite{janiak2023polysemantic}. -->

Upon deeper reflection, this should perhaps have been unsurprising. Like individual neurons—which may "privilege" computations that align with unit basis in their input space while not enforcing enforcing it—the nonlinearities implemented by individual attention heads might merely "privilege" head-aligned attention computations without strictly enforcing them.

It would therefore be ideal if our decomposition methods could cope with attention computations that are distributed across heads. So far, it has been difficult to find satisfactory activation-based decomposition methods that can do this <cite>jermyn2025attention</cite>. Fortunately, parameter decomposition methods offer some hope: Parameter components are vectors in parameter space, and therefore may span multiple attention heads. In fact, the parameter components found by VPD *usually* span multiple attention heads!

In this section, we study a behavior that probably every transformer language model exhibits: "*previous token behavior*", in which attention from timestep $t$ to $t-1$ carries forward information from the immediate past to the present. This behavior is typically associated with a particular type of attention head called a "*previous token head*" <cite>clark-etal-2019-bert,elhage2021mathematical, olsson2022incontextlearninginductionheads, wang2022interpretability</cite>. We show that such a head exists in our model, but show that previous token behavior is distributed across multiple heads. We find that a single pair of rank-one components, whose components span all heads in that layer, is responsible for a greater amount of previous token behavior than the network's "previous token head". We explore how "previous token behavior" is actually implemented in the model, and demonstrate through a combination of static analysis and interventions that "previous token behavior" is distributed across multiple attention heads.

**Identifying the model's "Previous Token Head"**

Like many language models, our model has a head that, on average, places the majority of its attention on the previous timestep (cref figure of attention scores for random tokens, appendix for attention scores for dataset tokens and for Wang et al. figure for comparison). In our model, this head is head 1 in layer 1, called **L1H1**. However, L1H1 is not the only head to assign substantial probability to the previous token; many other heads do too, including heads in the same layer as L1H1. We'll focus our analysis on the attention block in layer 1 for now.

<figure>
<label id="fig:prev_token_scores"/>
<img src="figures/prev_token_scores_combined.png">
<figcaption>Identifying the previous token head: Mean attention across multiple inputs on position $t-1$. **Left:** Average over sequences of random tokens. **Right:** Average over sequences sampled from the dataset. The plots reveal L1H1 is the most canonical "previous token head". But note other heads place substantial average attention on sequence position $t-1$.</figcaption>
</figure>

**Looking at parameter components in attention block 1**
 Let's take a look at a few parameter components in the layer where our previous token head lives. We'll focus on the components of the $W_Q$ and $W_K$ matrices. There are many interesting components that correspond to easily interpretable behaviors:

- Component XYZ activates on ABC todo
- Component ZYX activates on GHJ todo
- Component etc.

TODO graphs for multiple components

These findings are encouraging, because it suggests that our decomposition is finding parts of the network that are specialized for particular functional roles. There are two components, however, that seem not to be especially specialized. In fact, they have a mean CI of 0.9XX, meaning that they activate for almost every input!

TODO graphs for these two attention components specifically.

One of these is a component of the $W_Q$ matrix, and the other a component of the $W_K$ matrix. Interestingly, both seem to have the largest norm in L1H1, but have substantial weight norm in other heads, suggesting neither are exclusively "located" (i.e. have high weight norm) in any particular head (Figure <ref>fig:qk_comp_weight_norm</ref>). Since they are almost always active, they will therefore have an effect on the attention patterns at every sequence index at every head in layer 1.

<figure>
<label id="fig:qk_comp_weight_norm"/>
<img src="figures/layer1_qk_combined.png">
<figcaption>Caption todo</figcaption>
</figure>

It is possible to study the interaction between these components using a component-specific "QK circuit" <cite>elhage2021mathematical</cite>, which will give us a window to understanding their typical effect on the attention patterns of each head. If, within a head, they have a high inner product relative to other component pairs within that head, it indicates they will contribute a lot to that head's attention pattern. We can measure this quantitatively using a metric we call the "standardized attention contribution" between these two components:

<label id="eq:attn_contribution_head"/>
$$\text{AttentionContribution}(c, c', \tau, h) = \\
\Big(\text{sign}(\mathbb{E}_x\left[V_{Q,c}^\top x\right]) \cdot \lVert V_{Q,c} \rVert \cdot U_{Q,c}^h\Big)^\top \boldsymbol{R}_{\tau} \Big(\text{sign}(\mathbb{E}_x\left[V_{K,c'}^\top x\right]) \cdot \lVert V_{K,c'} \rVert \cdot U_{K,c'}^h\Big)$$
<!-- LaTeX original:
\label{eq:attn_contribution_head}
\text{AttentionContribution}(c, c', \tau, h) = \\
\Big(\text{sign}(\mathbb{E}_x\left[V_{Q,c}^\top x\right]) \cdot \lVert V_{Q,c} \rVert \cdot U_{Q,c}^h\Big)^\top \boldsymbol{R}_{\tau} \Big(\text{sign}(\mathbb{E}_x\left[V_{K,c'}^\top x\right]) \cdot \lVert V_{K,c'} \rVert \cdot U_{K,c'}^h\Big)
-->

Here $\boldsymbol{R}_{\tau}$ is the RoPE rotation matrix for offset $\tau$, which lets us understand the attention contribution between these components at different sequence position offsets. The attention contribution is a scalar that gives us a static indication (i.e. using almost no information about the data) of what effect a pair of components have on the attention pattern. These are not directly comparable across heads, which may have different scales and averages that nevertheless become irrelevant thanks to the Softmax. We therefore standardize the attention contribution:

<label id="eq:stand_attn_contr"/>
$$\text{StandardizedAttentionContribution}(c, c', \tau, h)= \frac{W(c, c', h, \tau) - \mu_h}{\sigma_h}$$
<!-- LaTeX original:
\label{eq:stand_attn_contr}
\text{StandardizedAttentionContribution}(c, c', \tau, h)= \frac{W(c, c', h, \tau) - \mu_h}{\sigma_h}
-->

where $\mu_h$ and $\sigma_h$ are the mean and standard deviation of the attention contributions across all $(c, c', \tau)$ for head $h$.
This standardization permits meaningful averages across heads.

<figure>
<label id="fig:attn_contrib_grid"/>
<img src="figures/layer1_qk_pair_lines_combined.png">
<figcaption>Caption todo</figcaption>
</figure>

We can see by comparing the sum of the attention contributions between all pairs (Figure <ref>fig:attn_contrib_grid</ref> – black lines) that it closely corresponds to the shape of the model's actual average attention logits (Figure <ref>fig:attn_patterns_layer1_random</ref>; see appdx todo for average attention plots on real data, which tell a similar story). This supports the idea that component interactions are a good way to decompose how attention at this layer actually works in the average case.

<figure>
<label id="fig:attn_patterns_layer1_random"/>
<img src="figures/layer1_attention_offset_profiles_random.png">
<figcaption>Caption todo</figcaption>
</figure>
 <!-- TODO Lee: I think this figure would be much better if we had the sum of the attention contributions plotted as well, so that we can more easily compare with the logits -->

The QK component pair that is always on (components *q.316* and *k.329*) has a high attention contribution in the first few timesteps of most of the six heads (Figure <ref>fig:attn_contrib_grid</ref>). Taken together means that this single component pair looks responsible for the majority of the attention to the recent sequence positions across all heads. This is borne out by the effects of interventions on the attention patterns: When we ablate only component *q.316*, the attention to the recent past is greatly reduced, whereas ablations of other Q components has minimal effect on average (Figure <ref>fig:attn_patterns_q_intv</ref>)

<figure>
<label id="fig:attn_patterns_q_intv"/>
<img src="figures/attn_q_L1_top10_n256_grid.png">
<figcaption>Caption todo</figcaption>
</figure>

Given that components *q.316* and *k.329* seem to be involved in generating most of the attention to the recent past, it begs the question: What are the attention *values* that their attention is carrying forward in time from the recent past? Are the different heads carrying forward distinct subspaces in the residual stream? After all, each head can carry only a small subspace of the full residual stream, and maybe it distributes previous token behavior across heads in order to carry forward a larger subspace of the residual stream.

We can quantify the extent to which $W_V^h$ matrices from different heads "read" from the same subspace using a metric we call the **subspace overlap** between the input spaces of the two matrices. To get the subspace overlap metric, we first calculate the Gram matrix for each head $h$:

$$M_h = {W_V^h}^\top W_V^h \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}.$$
<!-- LaTeX original:
M_h = {W_V^h}^\top W_V^h \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}.
-->

Intuitively, this matrix defines $W^h_V$'s "receptive field", since for any input vector $x$, the quantity $x^\top M_h x = \lVert W_V^h x \rVert^2$ measures how strongly $W^h_V$ amplifies that direction. We can leverage the fact that, for our Gram matrices,

$$\operatorname{tr}(M_a M_b) = \sum_{i,j} \lambda_i^a \lambda_j^b \bigl(\textbf{u}_i^a \cdot \textbf{u}_j^b\bigr)^2,$$
<!-- LaTeX original:
\tr(M_a M_b) = \sum_{i,j} \lambda_i^a \lambda_j^b \bigl(\textbf{u}_i^a \cdot \textbf{u}_j^b\bigr)^2,
-->

(where $\lambda_i^a$ are eigenvalues of $M_a$, i.e. the squared singular values of $W_V^a$) to give us a metric that is large when both heads have large eigenvalues in aligned directions. This quantity is unnormalized. We can normalize it using the Frobenius norm of both matrices, thus making the Frobenius cosine similarity between the two matrices (which is equivalent to the cosine similarity between the two Gram matrices viewed as vectors in $\mathbb{R}^{d_{\text{model}}^2}$). We can see that the different value projection matrices appear to read from different subspaces (Figure <ref>fig:attn_wv_overlap</ref> – Left).

<figure>
<label id="fig:attn_wv_overlap"/>
<img src="figures/layer1_wv_overlap_combined.png">
<figcaption>Caption todo</figcaption>
</figure>

However, not all directions in the residual stream are equally important! Data do not necessarily exist in all subspaces, or vary more in some than in others. We should therefore weight different dimensions according to the amount of data variance that exists along that axis. To do this, we form the **data-weighted** value matrix for each head. First, we mean-center the data and perform SVD to get the principal axes of variation:

$$\bar{X} = X - \bm{1} \bm{\mu}^\top, \qquad \bar{X} = \bar{U} \bar{S} \bar{Z}^\top,$$
<!-- LaTeX original:
\bar{X} = X - \bm{1} \bm{\mu}^\top, \qquad \bar{X} = \bar{U} \bar{S} \bar{Z}^\top,
-->

where $\bm{\mu} = \frac{1}{N}\sum_n \bm{x}_n$. We then rotate $W_V^h$ into the data's principal axes of variation and scale each axis by the corresponding singular value, yielding the data-weighted value projection matrix for head $h$:

$${W_V^h}_{\text{data}} = W_V^h \bar{Z} \bar{S} \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}.$$
<!-- LaTeX original:
{W_V^h}_{\text{data}} = W_V^h \bar{Z} \bar{S} \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}.
-->

We can apply the Frobenius cosine similarity in the same way. This metric suggests that, in general, value matrices in fact read from more similar subspaces, when the activations are taken into account (Figure <ref>fig:attn_wv_overlap</ref> – Right). In particular, L1H0, L1H1 (our previous token head), L1H2, and to some extent L1H3, all appear to attend to largely overlapping data subspaces. This may suggest some redundancy in the residual stream information being carried forward in time by the recent token behavior in these heads, though the fact that there is incomplete overlap suggests that some unique information is being carried forward. We did not notice any obvious semantic distinction between value components that tended to be activated by one head over another, and thus leave deeper investigation of this multi-head attention behavior, and others, to future work.

#### Case study 3: Bracket closing

<label id="sec:bracket-closing"/>


<label id="graph:bracket"/>

```graph
id: bracket-full
data: data/bracket-full.json
details: data/bracket-full-details.json
caption: Attribution graph for predicting `>` after `v` on the prompt `<` `u` `,` `v` `>`, pruned with adversarial sampling.<footnote>coefficient $0.5$ for cross-entropy reconstruction with stochastic sampling, coefficient $0.5$ for cross entropy with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.05$, $p=0.3$, $4000$ optimisation steps.</footnote> There are 158 active components in total. The probability on `>` is $1.000$ with ci masking, $0.997$ with stochastic masking and $0.363$ with adversarial masking under 4 PGD optimisation steps with step size 1. The target model assigned $0.547$ probability to `>`, indicating that this graph still isn’t quite capturing all the relevant computation going on. Though without adversarial the sampling we would be blind to this fact.
```

On the prompt `<` `u` `,` `v` `>` the target model correctly predicts that `>` follows `v`, assigning probability $0.538$. This requires remembering that there was an open angled brace earlier in the sentence that might be likely to close now. How does the model perform this task?

Figure <ref>graph:bracket</ref> shows an attribution graph for this prompt after adversarial pruning. We can see that information is carried over from the `<` sequence position to the `v` sequence position in the attention at layers $1,2$ and $3$. Here, we will first examine a single small computational pathway through the model on this prompt, much as we did for the princess example in the first case study. Then, we will slightly broaden our scope, and briefly survey all the attention components in the Figure <ref>graph:bracket</ref> graph that seem to be involved in transferring information about the opening bracket from the beginning of the prompt to the `v` sequence position.

<label id="graph:bracket_ci"/>

```graph
id: bracket-minimal
data: data/bracket-minimal.json
details: data/bracket-minimal-details.json
caption: Attribution graph for predicting `>` on the prompt `<` `u` `,` `v` `>`, pruned with causal importance masking.<footnote>coefficient $1.0$ for cross-entropy reconstruction with causal importance masking, importance minimality coefficient $0.1$, $p=0.3$, $2000$ optimisation steps.</footnote> There are 14 active components in total. The probability on `>` is $0.973$ with causal importance masking, $0.001$ with stochastic masking and $<0.001$ with adversarial masking under 4 PGD optimisation steps with step size 1. This indicates that the graph provides a very incomplete picture of the target model's computation for predicting `>`. Nevertheless, it may serve as a starting point for understanding the more complete graph in Figure <ref>graph:bracket</ref>.
```

**An oversimplified story**

Figure <ref>graph:bracket_ci</ref> shows a much smaller pruned graph of just $14$ nodes for this prompt. It correctly predicts `>` after `v` with very even higher probability than the target model on a causal importance masked forward pass, but fails completely under stochastic or adversarial masking. So it is clearly giving a very incomplete account of how the original model actually computes the bracket closing prediction on this prompt. Nevertheless, it might make for a good starting point for understanding the more complete graph in Figure <ref>graph:bracket</ref>.

Starting from the output, the largest direct positive attributions to `>` come from

1. A layer 3 mlp down projection component labeled "predicts closing angle brackets in text and markup". Looking at its activating examples and dataset attributions, it indeed seems to activate strongly primarily in contexts where the model may expect a right angle bracket to close at the next sequence position. Notably the component is sometimes not marked when it activates strongly. We speculate that this is because the component does not always end up influencing the model's final output prediction strongly even when it is active. This component in turn receives attribution from two components in the mlp input matrix labeled "fires inside angle brackets (html tags, irc nicks) to predict >" and "code and math identifiers predicting syntax symbols", with the latter appearing to fire on predictions for a more general set of closing delimiters.
2. A layer 2 mlp down projection component labeled "code, math, and legal text". It connects strongly to the two layer 3 mlp input components in addition to the output, suggesting that the layer 2 and 3 mlp components are not independent parallel pathways, but rather somewhat interlinked in series.
 <!-- Looking at the component's activating examples, it actually seems to primarily fire in contexts where the model may expect some sort of delimiter to close, such as \code{>}, \code{)}, \code,, \code{*}, \code{/}, \code{:}, or \code{.}, though the last one only in the context of abbreviations, not full stops in sentences. There are also some activations that superficially do not seem to fit this pattern, but might possibly be due to the model predicting delimiters mistakenly. Similar to the mlp 3 down projection component, it is also not always marked as causally important on all tokens between two brackets or other delimiters even when the component's activations are large in magnitude. -->
 This component in turn receives attribution from an mlp up projection matrix component labeled "syntactic punctuation and formatting in structured text".

These two mlp down projection matrix components also contribute the highest direct attribution to the output prediction `>` in the more complete graph in Figure <ref>graph:bracket_ci</ref>, and also in the original graph obtained from the SPD causal importance functions. Ablating them out of the target model itself, either independently or jointly, severely degrades performance on the `>` prediction, lowering the probability the model assigns from $0.538$ to $0.158$, $0.243$ for individual ablations and to $0.046$ for joint ablation. The model instead reassigns probability mass to other delimiters such as `)`, `_`, `,` or `)$`, suggesting that they are important for singling out a right angled bracket delimter in particular.

The mlp components in turn seem to receive the information indicating that there is an open angled bracket from the layer 2 attention output matrix components labeled "processing syntax in structured text like html and urls", and "html/xml tags and markdown formatting syntax", which receive it from a layer 2 attention value matrix component labeled "syntax tokens in markup and code" active on the `<` sequence position. This component in turn seems to receive the information directly from the `<` input embedding, as well as from a pathway routing through three layer 0 mlp components labeled "markup language syntax, urls, and structural delimiters", "fires on '<' and '</' to predict tag names" and "html/xml tag boundaries".

**A brief survey of attention components in the real graph**

The story told by the graph in Figure <ref>graph:bracket_ci</ref> might tempt us to think that the computation the model uses to predict the angled bracket closing is an extremely simple, perhaps even linear routing of information from input to output using a very small number of components specialized for this purpose. But a look at the graph in Figure <ref>graph:bracket</ref> obtained with adversarial pruning makes it clear that the actual computation is more intricate than that. While most of the components involved indeed appear to be quite specialized for predicting delimiter closing, or even angled bracket closing in particular, there are far more such components, spanning large subspaces within the model, than the simplified picture painted by Figure <ref>graph:bracket_ci</ref> might suggest.

We will not attempt to fully understand this graph here. But we will at least briefly survey the attention blocks which transmit the information from the `<` sequence position to the `v` sequence position:

The **layer 1 attention** has

- A single query component active on `v`, a bias which is causally important almost all of the time. This indicates that the relevant query at this layer is triggered as part of generic previous token behavior, not for any particular reason.
- Two key components active on `<`. One is labeled "punctuation, formatting boundaries, and newlines", the other is a broad bias that is almost always active. This indicates that the key information at this layer is sent from `<` partially as part of generic previous token behavior, and partially because `<` is a kind of formatting boundary.
- Eight value components active on `<`.
 Two are labeled "fires on punctuation and symbols" and "punctuation, syntax, and formatting tokens". They seem to be causally active on or predict punctuation, formatting, delimiters, newlines, symbols like `=`, various braces, white spaces, latex arrows and such. Notably, these same two components are also active on the `,` position.
 Another subcomponent is labeled "fires on angle brackets predicting html/xml tags" and is part of a cluster that also has two subcomponents active in the layer 2 attention value matrix. Subcomponents in this cluster all seem to be causally important primarily on various left angle brackets, `<`, `></ `, `} <` etc..
 <!-- Notably, all the components in it are part of value, key, and mlp matrices, never query or attention output matrices, which fits with the role of these two components in the graph as the source rather than destination of delimiter information. It also fits with the components in the cluster firing mostly on left angle brackets and comparably little on right angle brackets. -->
 Two more subcomponents ("fires on '<' in irc logs", "angle brackets (<, >) and inequality symbols") likewise fire primarily on angled brackets and related tokens. Another ("fires on message and speaker delimiters") fires on angled brackets and some other delimiters, such as `:` after `A` in the context of a Q&A. In all three cases, the subcomponent activations and causal importances tend to be much lower for closing angled brackets than opening angled brackets.
 Another subcomponent ("latex formatting and structural punctuation") fires on opening brackets more generally, including e.g. `{`, `[`, and variations like `\^ {`, as well as some delimiters like `;`, though apparently only in technical and math heavy contexts, and a few closing brackets like `);`. Again, the subcomponents' activation on these closing brackets is notably lower than on the opening brackets.
 Finally, one subcomponent only fires on the first token in a sequence.
- Five components in the attention output active on `v`.
 One is labeled "markup tags, latex macros, and irc usernames", and appears to be active primarily whenever an open left angled bracket (`<`, `.<`, etc.) has not been closed yet, or when the previous token was a backslash (`\`, `$\ `, etc.).
 Another, labeled "elements and separators in lists", seems to be active on and everywhere between separators and delimiters like commas or semicolons in lists, and various brackets in math or code.
 A third, "latex math syntax and code block formatting", seems to likewise activate primarily on tokens between delimeters, in this case seemingly exclusively various kinds of brackets in latex or code.
 The fourth, labeled "code and structured text syntax/indentation" appears to fire on any markup, HTML or other code and, seemingly to a somewhat lesser extent, on latex.
 Finally, one subcomponent, labeled "fires on names, citations, proper nouns and formatting tokens" was somewhat difficult for us make sense of. It fires on short text passages in succession, as if it is predicting something from the moment some left delimeter is seen until some other right delimiter is hit, but we could not determine from the examples what those delimiters are.
- There are also four subcomponents active on `,`: One bias key component that is always active, and three subcomponents related to punctuation, labeled "fires on commas and semicolons", ""punctuation, syntax, and formatting tokens", and "fires on punctuation and symbols". There is a single subcomponent active on `u`, labeled "fragments of proper nouns, foreign, and technical words".

Ablating the layers' attention output components out of the target model on the `v` sequence position degrades performance on the task severely, with the model now assigning $0.015$ probability on `<`. Its top logit instead becomes `<|endoftext|>` at $0.056$. Similarly, ablating the components out of the graph shown in figure <ref>graph:bracket</ref> reduces the probability on the `<` prediction down to $0.021$ under adversarial sampling. But the probability stays at $\approx 1.000$ under causal importance masking. This once again indicates that using naive masking schemes to infer causality can be very misleading, and adversarial sampling can help us avoid underestimating the number of components involved in the target model's computation.

The **layer 2 attention** has

- Two query components active on `v`, labeled "predicts punctuation, connectors, and sequence delimiters" and word fragments and prefixes anticipating word completions". They receive high positive attribution from both the layer 0 MLP down projection components and the layer 1 attention output components. Specifically, the latter subcomponent receives high positive attribution from the layer 1 attention output subcomponent labeled "markup tags, latex macros, and irc usernames" and a little from the one labeled "latex math syntax and code block formatting", which suggests that this query is partially triggered by the received closed angled bracket information from the layer 1 attention.
- Four key components active on the `<` sequence position.
 Two subcomponents fire on various opening brackets such as `<`, `(` and `[`, as well as other delimiters like opening quotation marks, `$` in latex, `**` and variations of these created by the tokeniser, like `[@`, `(*`, `_{`, `![ ` and such.
 The third fires only on the first sequence position of prompts, and on the `<|endoftext|>` token.
 The final one appears to be a "bias" that is almost always active.
- Nine value components active on the `<` sequence position.
 Two are part of the same "left angled brackets" cluster that also had an active subcomponent in the layer 1 attention value matrix at this same sequence position.
 Two others are part of another cluster of four components that seem to fire on left angled braces, but also left curly braces, opening quotation markers, and the start of links.
 The other five subcomponents likewise variously fire on left angled brackets, left brackets in general, left delimiters somewhat more generally, and in one case both left and right delimiters. Some of them are also causally important on the tokens after left delimiters as well, as though they are responding to the delimiters information being carried forward from the previous sequence position.
- Fourteen attention output components active on `v`.
 Two of them form a two component cluster and seem to fire whenever there are unclosed left delimiters, particularly many variations on left angled brackets, but also variations on left round, curly or boxy braces.
 Seven more seem to fire inside or on angled brackets or on other markup and xml related closing and syntax elements like e.g. `","`, '`[@`...`]`' and one appears to be active inside brackets in latex code.
 One subcomponent appears to be active on latex more generally, as well as some non-latex technical contexts like math and computer science writing. Another subcomponent is active on latex, but also on some code and foreign language text. One subcomponent is active inside angled brackets, but also on what appear to be chat messages, with particularly high magnitude activations on the line breaks in these messages. The final component is almost always active.


Ablating the attention layer 2 output out of the target model on the `v` sequence position severely degrades performance on the task, just as in attention layer 1. The model then still expects some kind of bracket, but not an angled bracket in particular. For example the probability on `)` goes up from $0.079$ to $0.279$, the probability on ` ] ` goes up from $0.015$ to $0.075$ and the probability on `);` goes up from $0.004$ to $0.052$. The probability on `>` goes down to $0.02$. This indicates that the information carried by the components in this attention layer is important for distinguishing which specific left bracket came before with sufficient confidence. As in attention layer 1, performance on the causal masked graph depicted in Figure <ref>graph:bracket</ref> remains essentially perfect even if we ablate these components. But performance is destroyed under adversarial masking, again confirming that finding a subset of components that is sufficient to compute the output is too permissive a criterion and can exclude component that play a significant computational role.

The **layer 3 attention** has

- One query component on the `v` sequence position, which seems to be almost always active.
- One value component active on the `v` sequence position, indicating that it is part of a self-attention mechanism in the graph. It is labeled "mathematical notation and formula component detector" and seems to be almost exclusively active on latex notation, though it also fires for some tokens that seem related to text that may typically feature latex, such as ` Eq`, ` Appendix`, ` proof`, and ` Newton`.
- Three value components active on the `v` sequence position, the first of which is also active on the `<` sequence position and labeled "syntactic glue, stopwords, and punctuation in varied text". It is causally important on more than $25\%$ of tokens, firing mostly on delimiters, "syntactic glue words" like ` and`, ` the`, ` a`, ` is`, ` would`, ` of`, ` on`, ` to` and to a lesser extent text following right after delimiters and these connective words.
 The two others are labeled "source code indentation and syntax" and "non-english and non-standard english text".
- Three attention output components. One is labeled "fires on non-english text", another "promotes latex mathematical commands and symbols
 ", and the third seems to be another "background" subcomponent, causally important on more than ninety percent of sequence positions.


This attention layer seems perhaps less crucial to the overall computation, though it still plays a role. Ablating two of the three attention output components out of the target model but keeping the bias component only lowers the probability it assigns to the `>` prediction down to $0.498$ from the original $0.547$. However, ablating the output bias component essentially destroys performance. We hypothesize that this is more due to the central role of this component in setting typical activation sizes, since it has very high attributions to many downstream nodes, than any sophisticated computational role. Notably, the same is not true of the layer two attention. Ablating the layer two attention output nodes out of the graph in Figure <ref>graph:bracket</ref> while keeping its bias node still reduces the probability on `>` under adversarial sampling down to less than $0.001$.

Notably, the query components used in layer 1 was a generic bias that is always active, and those used in layer 2 were "predicts syntax after nouns, numbers, and variables" and "word stem completion (stems to suffixes)". This somewhat begs the question of why the model would not predict `>` after `u` as well as after `v`. It turns out that it does in fact predict a closing angled bracket `>` as its top logit after `u` as well, though with lower confidence ($0.119$, whereas it assigns $0.547$ after `v`). Figure <ref>graph:bracket_u</ref> shows a graph of components involved in predicting `>` after `u`, generated with adversarial pruning. It is structurally similar to the graph in figure <ref>graph:bracket</ref>, with many of the same components being active. However, it is of course missing the components the latter graph has active in the attention layers on `u` and `,`, like "punctuation, symbols, and whitespace tokens" in the layer 1 value matrix on `,`. This suggests that the increased confidence of the model's prediction on `v` is due to the longer context further 'reinforcing' the possibility that the current context is about math and that a closing angled bracket is thus likely. Interestingly, the model does not predict a closing bracket after `,`, suggesting that it is aware that the comma suggests the statement in the bracket is not complete yet. We do not investigate the mechanisms underlying this heuristic here.


<label id="graph:bracket_u"/>

```graph
id: bracket-u-full
data: data/bracket-u-full.json
details: data/bracket-u-full-details.json
caption: Attribution graph for predicting "`>`" on the prompt `<` `u` `,` `v` `>` after "`u`", pruned with adversarial sampling.<footnote>coefficient $0.5$ for cross-entropy reconstruction with stochastic sampling, coefficient $0.5$ for cross entropy with $4$ steps of PGD, lr $1$, importance minimality coefficient $0.1$, $p=0.3$, $4000$ optimisation steps.</footnote> There are 162 active components in total. The probability on "`>`" is $0.996$ with ci masking, $0.997$ with stochastic masking and $0.092$ with adversarial masking under 4 PGD optimisation steps with step size 1. The target model assigned $0.119$ probability to "`>`". Comparing to Figure <ref>graph:bracket</ref>, many components occur in both graphs, particularly in the attention layers, suggesting similar computations occure in both graphs, though there are also substantial differences. This graph is of course lacking the components on "`u`" and "`,`" that seem to route additional information reinforcing the math context to "`v`" in Figure <ref>graph:bracket</ref>, which may account for the model's lower confidence in the "`>`" prediction earlier in the prompt.
```

Given how few components our decomposition has in total (ca. 10,000 alive in the whole model) it is perhaps remarkable how many of them appear to be dedicated to moving around and processing information for predicting closing delimiters of various kinds. This may be partially due to delimiter closing being one of perhaps relatively few prediction tasks simple enough for a model of this size to perform well at.

<!-- \subsection{Parameter components interact nonlinearly, but in an interpretable way} -->
<!-- Component interactions: geometry wrt neurons and coactivations. -->

## Early exploration: Characterizing nonlinear component interactions

<label id="sec:nonlin-comp-interactions"/>


In our case studies, we have tried to trace out the relationships between component activations in particular computations using attributions. This is not a complete account of how the model outputs are produced, since attributions only attempt to measure how strongly one component activation influences another. To fully reverse engineering neural networks with VPD, we will need some account of how component activations are actually computed from upstream component activations. In the case of MLP up projection matrices, as well as query, key and value matrices, we believe that this should not be difficult, because the connections to their preceding component activations in the graph are linear save for the layernorms. So we should be able to understand them almost entirely as linear combinations of preceding component activations.

In the case of MLP down projection and attention output matrices however, there are nonlinearities in the computational graph separating them from preceding component activations, neurons in the case of the MLP down projection and attention heads in the case of the attention output matrices. In the case of the MLP down projection components in particular, every component activation is a linear combination of many MLP neurons, each of which potentially connects to all MLP input component activations. One might thus worry that the nonlinear interactions between the input component activations that produce the output component activations could be inherently very complicated. At present, we cannot exclude this possibility, but we believe there are theoretical and empirical reasons to think that the interactions are much simpler than the raw number of nonlinearities in the neural network might suggest.

## Editing language model parameters by hand to modify its neural algorithm

<label id="sec:model-editing"/>


<!-- Mundane utility demo (e.g. steering, intervention, rewriting, editing in a conditional steering vector or unlearning memorized data, something else) -->
TODO. Sketch:
We use the decomposition for model editing. Specifically:

#### All emoticons are surprised face

<figure>
<img src="figures/editing_pareto.png"/>
<figcaption>Lora vs closed-form VPD based model editing results. TODO improve the y axis metric here</figcaption>
</figure>

```heatmap
left_data: data/editing-kl-heatmap-lora.json
left_title: LoRA
right_data: data/editing-kl-heatmap.json
right_title: VPD
caption: Per-token KL divergence after editing...
```

TODO. Sketch:

- The target model has multiple components in the mlp.2.down matrix that fire on the first token of emoticons. We picked one and edited its left singular vector, adding the unembedding vector of `o` to it (multiplied with a prefactor of 2). The resulting model always predicts that emoticons are surprised faces, without substantially altering its output distribution in non-emoji related contexts.
- See Figure ref for some examples of the old vs. new model output logits. Mean Off-target KL diff is xxx on these examples, and xxx on a random sample of xxx batches drawn from the whole dataset.
- Compared to LoRA fine-tuning, we get much fewer off-target effects for LoRAs trained on less than xxx examples.

## Discussion

<label id="sec:discussion"/>
TODO

**How much robustness is enough?**
 In principle, we would like our decomposition to be robust to every possible choice of partial ablation masks, because this ensures that our "local explanations" of the target model's behavior on individual data points, given by the causally important components, react as expected to model editing and can be arbitrarily aggregated into more "global explanations" of the network's behavior over multiple data points or the full data distribution.
For example, if we edited the target model by ablating some component or set of components out of the model weights, we could always be confident that the resulting new model would still give (approximately) the same outputs as the original model on all model inputs for which none of the ablated components were causally important. Or, for another example, suppose we analyzed how the target model predicts closing angled and rounded brackets on six different prompts, by studying the "subnetworks" given by the sums of causally important components for each individual prompt. Then we could take the sum of the causally important components for all six prompts, and obtain a 'union' subnetwork which still (approximately) outputs the same predictions for all six prompts.
<!-- This is important to us because it allows us to understand the target model incrementally. We can first find explanations for the model's behavior on individual data points, each involving small subsets of components, then aggregate them into larger sets of components that explain the model's behavior over sub-distributions (e.g. bracket closing tasks), then incrementally aggregate these sets of components with each other to slowly understand the target model's behavior under the full distribution. -->
But we have also pointed out a theoretical toy case in which strictly demanding adversarial robustness to all possible ablation masks would exclude decompositions we would intuitively regard as valid, because the adversary can systematically exploit random interference noise in 'unused' circuitry to change the network output (see Section <ref>sec:methods_adv</ref>). So, we want the decomposition be adversarially robust, but we do not actually want to demand that it is completely robust. How much robustness exactly do we want then?
We do not currently have a fully satisfying answer to this question. But we suggest that a reasonable approach may be to ground the answer in practical considerations: Which ablation masks correspond to sets of components we would actually encounter when attempting to understand or edit the model? And on which data points would we want to investigate the behavior of these sets of components? If none of the sets of components we analyse to understand the model over increasingly broad sub-distributions correspond to partial masking the decomposition is not robust to on any data point in these sub-distributions, and none of the model edits we want to make in practice correspond to ablation masks we are not robust to on any data point on which we care about the behavior of the edited model, then the lack of complete robustness may not be relevant to us in practice.
Even if we do encounter a partial ablation we are not robust to over the course of our investigations, the problems caused by this may be limited if they only apply to a few data points. For example, if editing the model by ablating a particular set of components should not change its behavior on any data point in a large sub-distribution $X_1$, but does in practice change its behavior on two data points $x_1, x_2 \in X_1$, then the edited model is still behaving as we would expect in the vast majority of cases.
<!-- This perspective also provides further motivation to care particularly about robustness to the same ablation scheme applied to many data points: In practice, we tend to care about the behavior of particular sets of ablation masks over multiple data points. -->
<!-- \paragraph{Components can explain model behavior at different levels of resolution} TODO It might be possible to train other causal importance functions for different levels of sparsity on the same components, allowing us to move to different points on the simplicty-reconstruction pareto frontier while keeping the same set of components. This might allow us to progressively understand models and model components at higher levels of detail, starting with very sparse causal importances that give simple but incomplete pictures of the forward pass, then moving to less simple but more accurate pictures of the same forward pass while retaining what we have already learned about the components. This way, we could understand the role of components incrementally, starting with the most crucial effect pathways on the output and slowly working our way up to a more complete understanding. -->

### Section todo: Just rough notes for now: Computational graph vs attribution graphs

<!-- Lee: Previously this was supposed to be the intro to the attr graph section, but it's being repurposed for the discussion. -->

In mechanistic interpretability, we want to understand *what* a network is doing and *how* it is doing it. Loosely speaking, this corresponds to understanding the network's representations ("what") and its computations ("how"). Typically, we study "what" a model is doing by studying their activations in order to get a sense of what the model is currently representing, and seeing how those representations change throughout the model. We can study activations using: Linear probes, SAE or crosscoder features,

<!-- Conceptually, in neural networks we view computations as "upstream" of representations. -->

Some of the main goals of mechanistic interpretability are to understand, on any given input, *what* a model is doing and *how* it is doing it.

By identifying individual computational units and studying their activations, we can get a sense of "what" a model is doing. In mechanistic interpretability, it would be nice to be able to use the network's raw "computational graph"—the directed acyclic graph that people use to define a network's operations, including its weights, matrix multiplications, activation vectors, and nonlinearities—in order to understand how it computes its output. If we could, then we could understand how the model actually implements its neural algorithm, letting us achieve goals like predicting its behavior off distribution, or modifying the parameters of the model in certain desirable ways.

Early mechanistic interpretability work did work with the network's raw computational graph: It studied the activations of individual neurons, and analyzed the raw parameters that connected neurons in one layer to neurons in the next (e.g. <cite>cammarata2020curve</cite>). But networks do not necessarily use individual neurons as their basic computational unit, leading to problems such as neuron polysemanticity. Ultimately their use caused interpretability more challenges than they solved, and motivated a number of other approaches, including CLTs and VPD.

<!-- Nonetheless, if mechanistic interpretability is to achieve its goals, it is not enough to identify computational nodes, like neurons, CLT latents, or parameter components. We also need a way to understand how they connect together in order to understand their interactions. -->

CLTs tackle the problem in a different way to studying individual neurons in the raw computational graph <cite>ameisen2025circuit</cite>. They identify latents to decompose and predict activation vectors, and where latents' activations are defined by a thresholded linear function of the activations. Interactions between latents are also therefore thresholded linear functions.
By freezing nonlinearities at their values used on a given forward pass, CLT can be used to make attribution graphs that gives an account of how information flows through the CLT, starting from the prompt, through intermediate CLT latents, and eventually to the output. The attribution between two nodes can be modelled as the sum of all the paths that activations took through the CLT nodes. This is a very human-interpretable account, and has had real benefits for increasing assurances for the reasons for safety-relevant model behaviors <cite>anthropic2026claudeopus46</cite>.
<!-- It also has additional benefits, such as the latents only interacting in series---not, for instance, in parallel at a given layer. -->

It is important to appreciate that CLTs are importantly different objects from the network's computation graph. CLTs instead learn a different graph, albeit one that has some attractive properties for interpretability. But it should be acknowledged that this was not a costless transition. It is not obvious that neural networks actually do operations that are well approximated using a thresholded linear activation function and linear interactions. Indeed, the fact that extremely large dictionary sizes are needed for CLTs to be good approximations of their target networks is suggestive of compensation for a mismatch in functional form. And while large enough CLTs can approximate the target network's computations to arbitrary accuracy, in the limit these large dictionaries begin to resemble large lookup tables, which are very unlikely to be parsimonious descriptions of the model's computations.

VPD, and other parameter decomposition methods, aim for their accounts to stay as close as possible to the network's computational graph, while also solving the issues with using the raw computational graph that the "individual neurons" approach encountered. The cost they pay for this is that the interactions between parameter components are not simply linear, as they are in CLTs. But what they gain is a greater claim to mechanistic faithfulness, and a relatively straightforward translation between mechanistic descriptions and objects in the network's computational graph. While CLTs give us a useful sense of "what" a model is doing, they make it harder to talk about "how" the model itself is doing is doing, because they take a step away from the model's computation graph. Parameter decomposition methods too can give us a sense of "what" a model is doing. And because they do not step away from the model's computation graph, there is an opportunity for further analysis to reveal the "how". We leave that analysis for future work.

But to fully leverage these benefits, we'll need to do more research to better characterize those nonlinear interactions between parameter components at a given layer. <!-- Lee: I'm not super happy with this section or this lead-in to attribution graphs. In part it feels like a good chunk of this section could go to the appendix. -->

It is also possible to construct attribution graphs for parameter components. At a high level, these are somewhat similar to CLT attribution graphs, in that we freeze nonlinearities and take the gradients at that point, thus getting a linear approximation of how information flows between parameter components on a particular prompt. But parameter component attribution graphs have some attractive properties over CLT attribution graphs:

- They involve components used in by the target model
- They natively describe attributions between components of any type, including attention.
- TODO some nice things to say about our graphs

<!-- In this paper, we don't aim to give a detailed account of the nonlinear interactions between components. We think such accounts are possible, -->

### Some other discussion subsection

### Related work

<label id="sec:related-work"/>


### Limitations

<label id="sec:limitations"/>


Things to maybe bring up here

- Our simplicity measures are imperfect and ad-hoc.
- We only show attribution graphs, not computational graphs. Attributions have a lot of issues and do not capture the full structure of non-linear interactions.

### Future work

<label id="sec:future-work"/>


### Conclusion



 <!-- or 'unsrt', or the specific neurips style -->

## Appendix

<label id="sec:app:appendix"/>


## Clustering Subcomponents into Parameter Components

<label id="app:clustering"/>

VPD decomposes each weight matrix $W_l$ into a sum of rank-one *subcomponents*: $W_l \approx \sum_{c} \vec{U^l_c} \vec{V_c^{l \top}}$. While each subcomponent parameterizes only a single weight matrix, a full *parameter component* should span the entire parameter space, potentially involving subcomponents from multiple weight matrices that work together to implement a single computation. We therefore need a method to identify which subcomponents across different weight matrices should be grouped together into coherent parameter components.

### Coactivation-Driven Clustering

We observe that subcomponents that participate in the same computation should activate together on the same datapoints. If subcomponent $c$ from layer $l$ and subcomponent $c'$ from layer $l'$ consistently have high causal importance values on the same inputs, they are likely implementing related computations and should be grouped into the same parameter component.

Let $g^l_c(x) \in [0, 1]$ denote the causal importance of subcomponent $c$ in layer $l$ on input $x$. Given a dataset $\mathcal{D} = \{x_1, \ldots, x_N\}$, we compute a *coactivation matrix* that measures how often pairs of subcomponents activate together:

$$\text{CoAct}_{i,j}
 = \sum_{x \in \mathcal{D}}
 \mathbf{1}[ g_i(x) > \tau ]
 \cdot \mathbf{1}[ g_j(x) > \tau ]$$
<!-- LaTeX original:
\text{CoAct}_{i,j}
 = \sum_{x \in \mathcal{D}}
 \mathbf{1}[ g_i(x) > \tau ]
 \cdot \mathbf{1}[ g_j(x) > \tau ]
-->

where $\tau$ is an activation threshold (we use $\tau = 0.01$ by default) and indices $i, j$ enumerate all subcomponents across all layers. The diagonal entry $\text{CoAct}_{i,i}$ gives the total activation count for subcomponent $i$.

### Minimum Description Length Clustering

We frame the clustering problem using the *Minimum Description Length* (MDL) principle [TODO: cite]. The goal is to find a grouping of subcomponents that minimizes the total cost of describing both the grouping structure and the activation patterns of the grouped components. Since the number of possible groupings grows according to Stirling numbers of the second kind, enumerating all partitions is infeasible. Instead, we use a stochastic merging approach guided by the MDL cost.

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

This cost has an intuitive interpretation: each time a group activates, we must encode (1) which group it is ($\log_2(k)$ bits) and (2) the rank-one matrices comprising that group ($\alpha \cdot r(P_i)$ bits, where $\alpha$ controls the penalty for group complexity).

### Stochastic Hierarchical Merging

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

where $r(P_{i,j}) = r(P_i) + r(P_j)$ is the rank of the merged group (approximated as the sum of the ranks of the individual groups [TODO: justify this approximation]).

Naively, one might greedily select the pair $(i^*, j^*) = \arg\min_{i < j} \Delta\mathcal{L}(P_i, P_j)$ and merge them. To allow exploration of alternative clusterings, we use stochastic selection: instead of always choosing the minimum-cost pair, we sample uniformly from all pairs within a threshold $\epsilon$ of the minimum cost.

### Choosing alpha

<!-- \todo{Shorten this and work it into the rest of the section} -->

Consider two rank-1 components $P_i, P_j$ with importances $s_i, s_j$, merged into a single component $P_{ij}$ with importance $s_{ij}$ and rank $r(P_{ij})=2$. We assume that $s_i(x), s_j(x)\in\{0,1\}$, which can be achieved by rounding causal importance values to $0$ or $1$ depending on whether they are below or above some cutoff. Then, pointwise, we have

$$s_{ij}(x) = s_i(x)\lor s_j(x) = s_i(x)+s_j(x)-s_i(x)s_j(x).$$
<!-- LaTeX original:
s_{ij}(x) = s_i(x)\lor s_j(x) = s_i(x)+s_j(x)-s_i(x)s_j(x).
-->

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
<!-- LaTeX original:
\label{eq:delta_simplified}
 &\Delta \MDL(P_i,P_j) \\
 &= (s_\Sigma - s_i - s_j)\log_2\!\Bigl(\tfrac{k-1}{k}\Bigr)
 + s_{ij}\log_2(k-1) - (s_i+s_j)\log_2(k)
 + \alpha(2s_{ij} - s_i - s_j).
-->

If we additionally approximate $\log_2(k-1)\approx \log_2(k)$ and neglect the small dictionary term
$\log_2\!\bigl(\tfrac{k-1}{k}\bigr)$,
this simplifies pointwise to

$$\Delta \mathcal{L}_{\text{MDL}}(x) \approx
 \alpha\bigl(s_i(x)+s_j(x)-2s_i(x)s_j(x)\bigr)
 -\log_2(k)\, s_i(x)s_j(x).$$
<!-- LaTeX original:
\Delta \MDL(x) \approx
 \alpha\bigl(s_i(x)+s_j(x)-2s_i(x)s_j(x)\bigr)
 -\log_2(k)\, s_i(x)s_j(x).
-->

Let the dataset have $X$ examples and write empirical averages as
$\mathbb{E}[\cdot] := \frac{1}{X}\sum_{x=1}^X(\cdot)$.
Averaging <ref>eq:delta_simplified</ref> then gives

<label id="eq:avg_delta"/>
$$\mathbb{E}[\Delta \mathcal{L}_{\text{MDL}}] \approx
 \alpha\bigl(\mathbb{E}[s_1]+\mathbb{E}[s_2]\bigr)
 -(2\alpha+\log_2 k)\,\mathbb{E}[s_1 s_2].$$
<!-- LaTeX original:
\mathbb{E}[\Delta \MDL] \approx
 \alpha\bigl(\mathbb{E}[s_1]+\mathbb{E}[s_2]\bigr)
 -(2\alpha+\log_2 k)\,\mathbb{E}[s_1 s_2].
 \label{eq:avg_delta}
-->

If $\mathbb{E}[s_1]=\mathbb{E}[s_2]$, then

$$\mathbb{E}[\Delta \mathcal{L}_{\text{MDL}}] \approx
 2\alpha\,\mathbb{E}[s_1]
 -(2\alpha+\log_2 k)\,\mathbb{E}[s_1 s_2].$$
<!-- LaTeX original:
\mathbb{E}[\Delta \MDL] \approx
 2\alpha\,\mathbb{E}[s_1]
 -(2\alpha+\log_2 k)\,\mathbb{E}[s_1 s_2].
-->

For binary indicators, we can write

<label id="eq:co_def"/>
$$\mathbb{E}[s_1 s_2]
 = \Pr(P_2\text{ important}\mid P_1\text{ important})\;\mathbb{E}[s_1].$$
<!-- LaTeX original:
\mathbb{E}[s_1 s_2]
 = \Pr(P_2\text{ important}\mid P_1\text{ important})\;\mathbb{E}[s_1].
 \label{eq:co_def}
-->

Denoting $\mathrm{co}(P_2\mid P_1):=\Pr(P_2\text{ important}\mid P_1\text{ important})$,
plugging <ref>eq:co_def</ref> into <ref>eq:avg_delta</ref> and canceling $\mathbb{E}[s_1]$ (assuming $\mathbb{E}[s_1]>0$) yields the threshold condition

$$
\begin{aligned}
0 &= 2\alpha - \mathrm{co}(P_2\mid P_1)\,(2\alpha+\log_2 k),
\end{aligned}
$$
<!-- LaTeX original:
0 &= 2\alpha - \mathrm{co}(P_2\mid P_1)\,(2\alpha+\log_2 k),
-->

so

$$\alpha
= \frac{\mathrm{co}(P_2\mid P_1)}{1-\mathrm{co}(P_2\mid P_1)}\cdot \frac{\log_2(k)}{2}.$$
<!-- LaTeX original:
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

<!-- The optimal permutation is found efficiently using the \todo{todo} on the confusion matrix between label assignments. Low distances across ensemble runs indicate stable, reproducible groupings that likely reflect genuine computational structure in the network. -->

## A training recipe for VPD

<label id="app:sec:recipe"/>
<!-- Training recipe: Which metrics matter and what (relative/absolute) values they should have -->

<!-- TODO(lee): We'll almost certainly have a brief two-liner in the main text about this and include this in the appendex essentially verbatim. -->

In this section, we offer practical guidance for applying VPD to other language models, based on our experience training with the model studied in this paper, as well as a range of other toy models. A file with default hyperparameter configurations can be found in the github repository ref todo.

**Evaluation metrics.**

To assess whether a VPD decomposition has converged to a satisfactory solution, we recommend tracking the following primary metrics:

1. **PGD reconstruction loss** (adversarial masks, freshly initialized): The most important metric. This evaluates reconstruction quality under adversarially chosen masks optimized independently for each batch. The setting we want is `per_batch_per_position`, see section <ref>sec:methods_adv</ref> for why. This is stricter than the persistent adversarial loss used during training and is our primary indicator of mechanistic faithfulness. For deeper models, more adversarial steps may be needed. As a rough heuristic, we keep $n_{\text{adv}} \cdot \text{lr}_{\text{adv}} \approx 2$; if increasing the number of steps, decrease the learning rate proportionally so the adversarial optimizer can fine-tune more precisely. For discussion on how much adversarial optimisation exactly our causal importances should be robust to, see Section <ref>sec:discussion</ref>.
2. **$L_0$ per data point**: The average number of subcomponents with nonzero causal importance on a data point. This should be tracked relative to the rank of the original weight matrices. For a transformer, MLP matrices typically have rank $d_{\text{resid}}$; the $L_0$ should be significantly smaller than this for the decomposition to be providing a useful simplification. Note that $L_0$ typically starts high and decreases steadily over training due to $p$-annealing (see below), so early in training the importance minimality loss value is a better predictor of what the final $L_0$ will be.

Additionally, we often monitor **Stochastic reconstruction loss**, because it indicates performance under the average permitted masking as opposed to worst-case maskings, **unmasked reconstruction loss** (all masks set to $1.0$, excluding the $\Delta$-components), because it indicates to what extent the sum of all components is identical to the target model even without the delta components and **CI-masked reconstruction loss** (using the causal importance values directly as masks) as well as **Rounded CI-masked reconstruction loss** (as CI-masked but all causal importance greater than zero are rounded to $1.0$) because they indicate performance when keeping exactly those components deemed causally important. Note though that the latter two are only useful indicators because VPD does not directly optimize for them: It would be (and in practice is) trivial to achieve almost perfect reconstruction for these two maskings if we included them in the training loss. But this would not indicate that our decomposition was actually capturing more of the target model's computation, because these metrics are not robust to "cheating" in the way the adversarial, and to a lesser extent stochastic reconstruction losses are.

**Training Loss terms.**

VPD training uses the following loss terms, each of which requires its own loss coefficient. We discuss considerations for tuning these below.

1. **Adversarial reconstruction loss** ($\mathcal{L}_{\text{PPGD recon}}$, coefficient $0.5$): This is the persistent PGD loss described in Section <ref>sec:methods_adv</ref>. Making the adversarial optimizer cheap yet effective is nontrivial. The adversarial learning rate usually needs to be tuned and depends on the regular learning rate. For the other hyperparameters of the adversarial optimiser, we recommend using the defaults described in Appendix <ref>app:methods</ref>: an Adam optimizer with $\beta_1 = 0.5$, $\beta_2 = 0.99$, constant learning rate with short warmup, and per-batch-per-position source scope. For smaller models, fewer adversarial steps per training step may suffice; for larger, especially deeper, models one may need more steps (and a correspondingly lower adversarial learning rate). We usually keep this loss coefficient fixed to $0.5$, setting the scale for the other losses.
2. **Stochastic reconstruction loss** ($\mathcal{L}_{\text{stochastic-recon}}$, coefficient $0.5$): This loss primarily prevents the optimization from stalling early in training, and secondarily prevents it from over-focusing on worst-case ablations at the expense of average-case reconstruction quality. We keep the coefficients for the two reconstruction losses equal and normalized to $\frac{1}{2}$ each. We usually keep this loss coefficient fixed to $0.5$, setting the scale for the other losses.
3. **Importance minimality loss** ($\mathcal{L}_{\text{importance-minimality}}$): This is typically one of the most sensitive hyperparameters and often requires tuning. The $p$-norm exponent is annealed linearly from $p_0 = 2.0$ to $p_{\text{final}} = 0.4$ over the full training run (with $\beta = 0.5$ for the frequency term). We recommend keeping this annealing schedule fixed and tuning the coefficient instead. Setting it too high leads to collapsed decompositions with poor reconstruction; too low leads to decompositions where too many components are simultaneously active.
4. **Frequency minimality loss** ($\mathcal{L}_{\text{frequency-minimality}}$): The coefficient for this term also requires some tuning, but interacts with the importance minimality coefficient: increasing the frequency penalty effectively increases sparsity pressure, so it may be necessary to lower the importance minimality coefficient to compensate. As a starting point, we suggest setting the frequency penalty coefficient at roughly $0.5\times$ the importance minimality coefficient, unless problems are observed. Too low a coefficient tends to produce fewer, overly polysemantic components.
5. **$\Delta$-component L2 penalty** ($\mathcal{L}_{\text{Delta-L2}}$): This penalizes the MSE between the sum of subcomponents and each target weight matrix. In practice, this coefficient is not very sensitive. We recommend increasing it by factors of $10$ from a conservative starting point until the unmasked reconstruction loss becomes negligibly small. It is safe to overshoot the coefficient considerably, though making it excessively large can still impair optimization.

**Subcomponent count $C$.**
 The number of rank-one subcomponents per weight matrix is not extremely sensitive. It should be set large enough for the optimization to capture all the components that are present. If unsure, we recommend erring on the side of too many subcomponents, then inspecting the spectrum of log mean causal importances (averaged over a batch) at the end of an exploratory run. There is typically a sharp cutoff in this spectrum separating "alive" from "dead" subcomponents, which reveals how many components are actually in use. The optimization tends to work best when $C$ is larger than needed, but not excessively so—roughly within a factor of $2$ of the true number of components appears to work well.

**Causal importance function**
 For decomposing transformer models, we recommend using `global_shared_transformer` as the causal importance function. This is itself a transformer model, which receives the concatenated hidden activations of the target model as input, and produces causal importances for all components as output. We typically choose the depth of this transformer to be within $\frac{1}{2}-2$ times the depth of the target model, though we have not investigated this hyperparameter as much as some others. We choose the residual stream to be wider than that of the target model since it needs to accommodate all of its hidden activations. For this paper, we used $2048$ compared to $768$ for the target model. As is somewhat standard, we usually choose the MLP width to be ca. four times the width of the residual stream.

**Summary**
 Applying the method to a new model usually requires adjusting

1. The importance minimality loss coefficient.
2. The learning rate
3. The adversarial learning rate
4. The frequency penalty loss coefficient
5. The number of components $C$
6. The Delta L2 penalty loss coefficient.

In our experience, the first three typically require the most extensive tuning. For larger models, the size of the model used for the causal importance function will likely need to be increased as well. The number of adversarial steps and the adversarial learning rate may also require adjustment.

## VPD Method Details

<label id="app:methods"/>
<!-- TODO: Full discussion of training details -->
<!-- - Adversarial reconstruction loss -->
<!-- - Importance function architecture -->
<!-- - P-annealing schedule -->
<!-- - Frequency penalty -->
<!-- TODO: Full discussion of attributions -->

<!-- TODO: Some discussion of autointerp -->

### Losses

#### Auxiliary delta component L2 penalty

<label id="app:delta_l2"/>
The Delta-components are different from normal subcomponents we train. Their rank can be greater than $1$, meaning they can be more complicated objects than regular subcomponents. We thus have a particular interest in ensuring that they are not used to compute the model's outputs. Theoretically, since we define the causal importances of Delta-components to always be zero, the stochastic and adversarial losses should ensure that this is the case. But in practice our reconstruction losses are not perfect, so we additionally encourage the Delta-components to be exactly zero with an auxiliary loss

$$\mathcal{L}_{\text{Delta-L2}}=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}\left(\Delta^l_{i,j}\right)^2=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}{\left( W^{l}_{i,j}- \sum^C_{c=1} U^l_{i,c} V^l_{c,j}\right)}^2.$$
<!-- LaTeX original:
\mathcal{L}_{\text{Delta-L2}}=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}\left(\Delta^l_{i,j}\right)^2=\frac{1}{N}\sum^L_{l=1}\sum_{i,j}{\left( W^{l}_{i,j}- \sum^C_{c=1} U^l_{i,c} V^l_{c,j}\right)}^2.
-->

#### Stochastic reconstruction loss

<label id="app:subset_recon"/>
<cite>bushnaq2025spd</cite> found that using a reconstruction loss which samples stochastic masks

$$\begin{aligned}
&m^l_c(x,t,r_\text{stoch}):=g^l_c(x,t)+\left(1-g^l_c(x,t)\right)r^l_{\text{stoch},c}(x,t)\\
&r^{l}_{\text{stoch},c}(x,t) \sim \mathcal{U}(0,1)
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
&m^l_c(x,t,r_\text{stoch}):=g^l_c(x,t)+\left(1-g^l_c(x,t)\right)r^l_{\text{stoch},c}(x,t)\\
&r^{l}_{\text{stoch},c}(x,t) \sim \mathcal{U}(0,1)
\end{aligned}
-->

for all target model matrices $l$ simultaneously<footnote>Simply called "stochastic reconstruction loss" in that paper, but here we reserve that term for the formulation that ends up in the training loss</footnote>

$$\begin{aligned}
\mathcal{L}_{\text{stochastic-recon-all}}&=D \left( f\left(x\vert {W'}^1\left(x,t,r^{\text{stoch}}\right),\dots, {W'}^L\left(x,t,r^{\text{stoch}}\right)\right),f\left(x\vert W^1,\dots,W^L\right) \right) \\
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
\mathcal{L}_{\text{stochastic-recon-all}}&=D \left( f\left(x\vert {W'}^1\left(x,t,r^{\text{stoch}}\right),\dots, {W'}^L\left(x,t,r^{\text{stoch}}\right)\right),f\left(x\vert W^1,\dots,W^L\right) \right) \\
\end{aligned}
-->

together with a layer-wise stochastic reconstruction loss which samples stochastic masks for one target model matrix at a time

<label id="eq:layerwise_random_recon"/>
$$\begin{aligned}
\mathcal{L}_{\text{stochastic-recon-layerwise}}=\frac{1}{L}\sum^L_{l=1} D \left( f\left(x\vert W^1,\dots,W'^l(x,t,r^{\text{stoch}}),\dots,W^L\right),f\left(x\vert W^1,\dots,W^L\right) \right)
\end{aligned}$$
<!-- LaTeX original:
\label{eq:layerwise_random_recon}
\begin{aligned}
\mathcal{L}_{\text{stochastic-recon-layerwise}}=\frac{1}{L}\sum^L_{l=1} D \left( f\left(x\vert W^1,\dots,W'^l(x,t,r^{\text{stoch}}),\dots,W^L\right),f\left(x\vert W^1,\dots,W^L\right) \right)
\end{aligned}
-->

performed better than training either $\mathcal{L}_{\text{stochastic-recon-all}}$ or $\mathcal{L}_{\text{stochastic-recon-layerwise}}$ alone, due to covering a somewhat more structurally diverse set of ablation. However, layer-wise reconstruction loss requires one forward-pass for every matrix in the model we decompose, which is expensive. For VPD training, we unify $\mathcal{L}_{\text{stochastic-recon-all}}$ and layerwise stochastic reconstruction loss $\mathcal{L}_{\text{stochastic-recon-layerwise}}$ into a single stochastic reconstruction loss. For every sequence position and batch index, we independently sample a number $\in\{1,\dots,L\}$, where $L$ is the number of weight matrices in the target model. We draw that many of the target model's weight matrices, sample stochastic masks for only those, and perform a forward pass replacing those matrices with the masked ones. This is no more computationally expensive than $\mathcal{L}_{\text{stochastic-recon-all}}$, and covers more structurally diverse ablations than layer-wise stochastic reconstruction losses, since it includes subsets of single matrices as well as the whole set as special cases.

#### Adversarial reconstruction losses

The optimization objective of the adversarial optimiser is maximizing the reconstruction loss on the masked forward pass:

$$\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}:=\sum_x D \left( f\left(x\vert W'^1(x,t,r^{{\text{adv}},1}),\dots,W'^L(x,t,r^{{\text{adv}},L})\right),f(x\vert W^1,\dots, W^L) \right)
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
\mathcal{L}_{\text{adversarial-recon}}:=\sum_x D \left( f\left(x\vert W'^1(x,t,r^{{\text{adv}},1}),\dots,W'^L(x,t,r^{{\text{adv}},L})\right),f(x\vert W^1,\dots, W^L) \right)
\end{aligned}
-->

by optimising adversarial sources $r^{{\text{adv}},l}(x,t)$ for the masks $m^l_c(x,t,r^{\text{adv}})$:

$$\begin{aligned}
m^l_c(x,t,r^{\text{adv}}) &:=g^l_c(x,t)+(1-g^l_c(x,t))r^{\text{adv},l}_c(x,t)\\
W'^l_{i,j}(x,t,r^{{\text{adv}},l})&:=\sum^C_{c=1} U^l_{i,c} m^l_c(x,t,r^{\text{adv}}) V^l_{c,j}
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
m^l_c(x,t,r^{\text{adv}}) &:=g^l_c(x,t)+(1-g^l_c(x,t))r^{\text{adv},l}_c(x,t)\\
W'^l_{i,j}(x,t,r^{{\text{adv}},l})&:=\sum^C_{c=1} U^l_{i,c} m^l_c(x,t,r^{\text{adv}}) V^l_{c,j}
\end{aligned}
-->

of component $c$ of target model matrix $l$ on batch index $x$ at sequence position $t$. The optimiser we use is with projected gradient ascent, where the projection clamps the sources $r^{{\text{adv}},l}(x,t)$ to the interval $[0,1]$ at every update step to ensure that the masks $m^l_c(x,t,r^{\text{adv}})$ stay between $0$ and $1$.
The sources for the Delta components' masks (see Section <ref>sec:method_components</ref>) are treated identically to the regular components, i.e. they are also adversarially optimized.

**PPGD adversarial reconstruction losses for training**

For training, we optimise a single set of sources $r^{\text{adv},l}_c(x,t)$ that persists across batches, with $x$ ranging across the batch index and $t$ across sequence position. On every batch, the adversarial AdamW optimiser performs $n_{\text{adv}}$ update steps on the adversarial sources $r^{\text{adv},l}_c(x,t)$, trying to maximise the adversarial loss $\mathcal{L}_{\text{adversarial-recon}}$ (In this paper, we used $n_{\text{adv}}=2$).

<!-- The adversarial optimiser also supports different scopes for the sources $r^l(x,s)$. \code{per\_batch\_per\_position} uses one unique source for every batch element and sequence position, \code{repeat\_across\_batch} uses the same source $r^l(x)$ for every batch element, but different sources for different sequence positions, and \code{single\_source} uses one unique source $r^l$ for every component. We use \code{per\_batch\_per\_position} in our experiments because it seems to perform best in practice. -->
<!-- \begin{equation}\label{eq:PPGD_recon} -->
<!-- \begin{aligned} -->
<!-- \mathcal{L}_{\text{PPGD recon}}&=\sum_x D \left( f\left(x\vert W'(x,s,r^{l, \text{adv}}\right),f(x\vert W) \right) \\ -->
<!-- \end{aligned} -->
<!-- \end{equation} -->

<!-- The VPD AdamW optimiser performs one update step on the subcomponents and the parameters of the causal importance function by calculating the gradient of the overall VPD loss function components and their causal importance functions. -->
<!-- \begin{equation} -->
<!-- \mathcal{L}_{\text{VPD}}:=\mathcal{L}_{\text{stochastic-recon}}+\mathcal{L}_{\text{adversarial-recon}}+\mathcal{L}_{\text{importance-minimality}}+\mathcal{L}_{\text{frequency-penalty}}+\mathcal{L}_{\text{Delta-L2}}. -->
<!-- \end{equation} -->

**PGD adversarial reconstruction loss for evaluation**

Continuously updating a single set of persistent adversarial sources is more computationally efficient, but not principled. Hypothetically, the VPD optimiser might trap the adversarial optimiser in some local extremum at some point during training, rendering the adversarial loss useless. Thus, for evaluation, we instead use a new randomly set of adversarial sources $r^{l, \text{adv}}$ for every batch, but increase the number of adversarial optimisation steps per batch $n_{\text{adv}}$.
<!-- Just as with the PPGD loss, the code supports \code{per\_batch\_per\_position}, \code{repeat\_across\_batch} and \code{single\_source} scope. -->
We use a single fixed $r^{\text{adv},l}_c$ for every sequence position $s$ and batch index $x$, to prevent the adversarial optimiser from fine-tuning to noise on particular data points. See section <ref>sec:methods_adv</ref> for more discussion.

### Frequency penalty motivation

<label id="app:frequency_penalty"/>
Here, we argue that subcomponents that are causally important more frequently effectively need to be specified to more bits of precision, with the extra description length cost in bits scales roughly with the logarithm of the frequency.

In the idealized setting, subcomponents are vectors of real numbers. We instead store them as vectors of finite precision floats. This quantisation effectively induces a discrepancy $\delta^l_c$ in parameter space between the ideal subcomponent, and our floating point approximation of the subcomponent. At sufficiently high float precision, the expected size of this discrepancy will scale as $\approx a_1 2^{-b^l_c}$, where $b^l_c$ is a bit count and $a_1$ is some constant.
Suppose we want to keep the impact of this discrepancy on our decomposition low. Specifically, we want the number of bits $b^l_c$ to be large enough for the KL divergence between the VPD forward pass outputs and the target model forward pass outputs summed over a dataset to stay below some fixed $\epsilon>0$. How large will we need to make $b^l_c$ as a function of $\epsilon$ to achieve this?

Over a dataset of $X$ inputs of sequence length $T$, a subcomponent will be causally important with some frequency $\frac{\sum_{x,t}\vert g^l_c(x,t)\vert^0}{X T}$. As a simplified toy model, let us assume that applying some small perturbation of size $\delta$ along the direction of the component in parameter space does not change the model output at all on data points where $g^l_c(x,t)=0$, but increases the KL divergence to the original model outputs by some $h(\delta)$ on data points where $g^l_c(x,t)=1$, where $h$ is an analytic function that is approximately the same for every component and every data point. Then, the increase to the total loss summed over all $X T$ data points from adding a perturbation $\delta$ to component $c$ is of approximate size $\approx \sum_{x,t}\vert g^l_c(x,t)\vert^0 h(\delta)$. This yields the inequality

$$\begin{aligned}
&\log_2(h(\delta))+\log_2(\sum_{x,t}\vert g^l_c(x,t)\vert^0)<\log_2(\epsilon)\\
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
&\log_2(h(\delta))+\log_2(\sum_{x,t}\vert g^l_c(x,t)\vert^0)<\log_2(\epsilon)\\
\end{aligned}
-->


Since $h$ is an analytic function, for sufficiently small $\delta$, it can be Taylor approximated to leading order as $a_2 \delta^n$ with some $n\in\{1,2,\dots\}$ for sufficiently. Inserting this approximation yields:

$$\begin{aligned}
b^l_c&>\frac{1}{n}\log_2(\sum_{x,t}\vert g^l_c(x,t)\vert^0)-\frac{\log_2(\epsilon)}{n }+\frac{\log_2(a_2)}{n}+\log_2(a_1)\\
\end{aligned}$$
<!-- LaTeX original:
\begin{aligned}
b^l_c&>\frac{1}{n}\log_2(\sum_{x,t}\vert g^l_c(x,t)\vert^0)-\frac{\log_2(\epsilon)}{n }+\frac{\log_2(a_2)}{n}+\log_2(a_1)\\
\end{aligned}
-->


So, the required bit precision $b^l_c$ grows approximately linearly with the logarithm of the component activation count across the dataset.

### Causal Importance Function Architecture

<label id="app:ci_function"/>

The causal importance function $\Gamma$ maps the target model's hidden activations to
per-component causal importances. It is a single, shared network that jointly computes causal
importances for all components across all weight matrices in the target model.

**Inputs.**

Let $L$ denote the number of weight matrices being decomposed, and let $a^l(x,t) \in
\mathbb{R}^{d_l}$ denote the input hidden activation to weight matrix $l$ of the target model at batch element $x$ and
sequence position $t$. Each activation vector is independently RMS-normalized, and the
normalized vectors are concatenated to form the input:

$$a(x, t) = \left[
 \operatorname{RMSNorm}(a^1(x,t)) \;\|\; \cdots \;\|\;
 \operatorname{RMSNorm}(a^L(x,t))
 \right] \in \mathbb{R}^{D},
 \quad D = \sum_{l=1}^{L} d_l.$$
<!-- LaTeX original:
a(x, t) = \left[
 \operatorname{RMSNorm}(a^1(x,t)) \;\|\; \cdots \;\|\;
 \operatorname{RMSNorm}(a^L(x,t))
 \right] \in \mathbb{R}^{D},
 \quad D = \sum_{l=1}^{L} d_l.
-->

**Input projection.**

The concatenated activation vector is linearly projected to the transformer's model dimension:

$$h^{(0)}(x, t) = W_{\mathrm{in}} a(x, t) + b_{\mathrm{in}},
 \quad W_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}} \times D}, \;
 b_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}}}.$$
<!-- LaTeX original:
h^{(0)}(x, t) = W_{\mathrm{in}} a(x, t) + b_{\mathrm{in}},
 \quad W_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}} \times D}, \;
 b_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}}}.
-->

**Transformer blocks.**

The projected activations are processed by $N$ pre-norm transformer blocks. Each block $n \in
\{1, \ldots, N\}$ applies bidirectional multi-head self-attention followed by a feedforward
network, each with a residual connection:

$$
\begin{aligned}
\hat{h}^{(n)}(x, t) = h^{(n-1)}(x, t) +
\operatorname{Attn}\!\left(
\operatorname{RMSNorm}\!\left(h^{(n-1)}(x, \cdot)\right)
\right)\!(t), \\
h^{(n)}(x, t) = \hat{h}^{(n)}(x, t) +
\operatorname{FFN}\!\left(
\operatorname{RMSNorm}\!\left(\hat{h}^{(n)}(x, t)\right)
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

After the final transformer block, a linear output head projects back to the total number of
components:

$$\Gamma(a(x,t)) = W_{\mathrm{out}} h^{(N)}(x,t) + b_{\mathrm{out}},
 \quad W_{\mathrm{out}} \in \mathbb{R}^{C_{\mathrm{total}} \times d_{\mathrm{model}}}, \;
 C_{\mathrm{total}} = \sum_{l=1}^{L} C_l.$$
<!-- LaTeX original:
\Gamma(a(x,t)) = W_{\mathrm{out}} h^{(N)}(x,t) + b_{\mathrm{out}},
 \quad W_{\mathrm{out}} \in \mathbb{R}^{C_{\mathrm{total}} \times d_{\mathrm{model}}}, \;
 C_{\mathrm{total}} = \sum_{l=1}^{L} C_l.
-->

The output is partitioned according to each matrix's component count $C_l$.

### Leaky hard sigmoids

<label id="app:sigmoids"/>
Theoretically, the causal importance for component $c$ in matrix $l$ is obtained simply by clamping the outputs of the causal importance function $\Gamma(a(x,t)) $ to the interval$[0,1]$ with a hard sigmoid function:

$$g^l_c(x, t) = \sigma_{\mathrm{h}}\!\left(\Gamma(a(x,t))^l_c\right),
 \quad \sigma_{\mathrm{h}}(z) = \mathrm{clamp}(z, 0, 1),$$
<!-- LaTeX original:
g^l_c(x, t) = \sigma_{\mathrm{h}}\!\left(\Gamma(a(x,t))^l_c\right),
 \quad \sigma_{\mathrm{h}}(z) = \mathrm{clamp}(z, 0, 1),
-->

However, in practice, the flat regions in a hard sigmoid function can lead to dead gradients for inputs below $0$ or above $1$. To avoid this, we use leaky hard sigmoids instead.
Specifically, we use *lower-leaky* hard sigmoids $\sigma_{H,\text{lower}}(x)$ for the causal importance used to create the masks for the actual forward passes for the $\mathcal{L}_{\text{stochastic-recon}}$ and $\mathcal{L}_{\text{stochastic-recon-layerwise}}$ losses, and we use *upper-leaky* hard sigmoids $\sigma_{H,\text{upper}}(x)$ in the importance minimality loss $\mathcal{L}_{\text{importance-minimality}}$ and the frequency penalty $\mathcal{L}_{\text{frequency-minimality}}$:.

The lower-leaky hard sigmoid $\sigma_{H,\text{lower}}(x)$ has a forward pass identical to a regular hard sigmoid, but below $0$, it uses a straight-through gradient estimator: Gradients pass through for $z \leq 0$ scaled by a leak coefficient $\alpha = 0.01$ when the incoming gradient is negative, preventing components from becoming permanently deactivated. The upper-leaky hard sigmoid is identical to a regular sigmoid for $z \leq 1$, but has a slope of $0.01$ above $1.0$, to avoid

We use a straight-through estimator for the lower leaky hard sigmoid instead of actually modifying the slope on the forward pass to avoid creating subcomponent masks smaller than zero. We restrict the straight-through estimator to apply only to negative gradients to prevent entries of $\Gamma(a(x,t))^l_c$ from updating to become ever more negative indefinitely.

This is in contrast to <cite>bushnaq2025spd</cite>, where the lower leaky hard sigmoid did have an actual slope of $0.01$ below $0$ on the forward pass. We made this change because we discovered that negative masks actually led to instabilities. For example, we found that the spurious component splitting observed for too-high importance minimality loss coefficients depicted in Figure 8 of that paper largely disappears if the straight-through estimator is used instead.

**Hyperparameters.**

Table <ref>tab:ci-hyperparams</ref> lists the hyperparameters used for $\Gamma$ in our experiments.
The target model is a 4-layer Llama-style transformer with $d_{\mathrm{model}} = 768$ and
$d_{\mathrm{intermediate}} = 3072$, decomposed across $L = 24$ weight matrices
(6 per layer: `c_fc`, `down_proj`, `q_proj`, `k_proj`,
`v_proj`, `o_proj`), yielding a total of $C_{\mathrm{total}} = 39,936$
components and an input dimension of $D = 27,648$.

<label id="tab:ci-hyperparams"/>
| **Parameter** | **Value** |
|---|---|
| CI model dimension ($d_{\mathrm{model}}$) | 2048 |
| Transformer blocks ($N$) | 8 |
| Attention heads | 16 |
| Head dimension | 128 |
| FFN hidden dimension ($d_{\mathrm{ff}}$) | 8192 |
| Positional encoding | RoPE (base $= 10,000$, max length $= 512$) |
| Attention | Bidirectional (no causal mask) |
| Activation function | Leaky hard sigmoid ($\alpha = 0.01$) |
*Hyperparameters for the causal importance function $\Gamma$.*

### p-annealing

The $L^p$ quasi-norm in the importance minimality loss $\mathcal{L}_{\text{importance-minimality}}$ and frequency penalty $\mathcal{L}_{\text{frequency-minimality}}$ (Equations <ref>eq:minimal</ref> and <ref>eq:freq_minimality</ref>) serves as a
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

## Interaction graphs




### Gradient attributions

 <label id="app:gradient_attributions"/>

 To understand how components interact with each other during the forward pass, we compute gradient
 attributions between pairs of subcomponents at adjacent layers in the computational graph. These
 attributions form the edges of an interaction graph that visualizes the flow of information through
 the decomposed model on a given prompt or aggregated over the dataset.

 Recall that each subcomponent $c$ at layer $l$ has a *component activation* $a^l_c(x,t) =
 {V^l_c}^\top h^l(x,t)$, where $h^l(x,t)$ is the pre-weight activation at layer $l$ on datapoint $x$
 at sequence position $t$. This is the projection of the input onto the right singular vector of the
 rank-one subcomponent, and it determines how strongly the subcomponent contributes to the layer's
 output.

 For a source subcomponent $c_s$ at layer $l_s$ and a target subcomponent $c_t$ at layer $l_t$ (where
 $l_s$ feeds into $l_t$ in the computational graph), we define the *gradient attribution* on
 datapoint $x$ at source position $t_s$ and target position $t_t$ as:


$$\alpha(c_s \to c_t; x, t_s, t_t) = \frac{\partial a^{l_t}_{c_t}(x,t_t)}{\partial
 a^{l_s}_{c_s}(x,t_s)} \cdot a^{l_s}_{c_s}(x,t_s) \cdot g^{l_s}_{c_s}(x,t_s)$$
<!-- LaTeX original:
\alpha(c_s \to c_t; x, t_s, t_t) = \frac{\partial a^{l_t}_{c_t}(x,t_t)}{\partial
 a^{l_s}_{c_s}(x,t_s)} \cdot a^{l_s}_{c_s}(x,t_s) \cdot g^{l_s}_{c_s}(x,t_s)
-->


 where $g^{l_s}_{c_s}(x,t_s)$ is the causal importance of the source subcomponent. The gradient
 $\times$ activation product $\frac{\partial a^{l_t}_{c_t}}{\partial a^{l_s}_{c_s}} \cdot
 a^{l_s}_{c_s}$ gives a first-order estimate of how much the source subcomponent's activation
 contributes to the target subcomponent's activation. Weighting by the causal importance
 $g^{l_s}_{c_s}(x,t_s)$ ensures that subcomponents which are not causally important on a given
 datapoint (i.e., those that can be ablated without affecting the output) do not contribute to the
 attribution, even if they happen to have nonzero activations and gradients.

 For most adjacent layer pairs, source and target positions coincide ($t_s = t_t$), since MLP and
 attention projection layers operate position-wise. However, for edges from key or value subcomponents
 to attention output subcomponents within the same attention block, the source position $t_s$ can be
 any position up to and including the target position $t_t$ (i.e., $t_s \leq t_t$, respecting the
 causal attention mask). This reflects the fact that key and value activations at earlier positions
 influence the attention output at later positions.



**Dataset-aggregated attributions.**

 To obtain a summary of how components interact across the dataset, we aggregate attributions over all
 datapoints and all valid position pairs:


$$A(c_s \to c_t) = \sum_{x \in \mathcal{D}} \sum_{t_s, t_t} \frac{\partial
 a^{l_t}_{c_t}(x,t_t)}{\partial a^{l_s}_{c_s}(x,t_s)} \cdot a^{l_s}_{c_s}(x,t_s) \cdot
 g^{l_s}_{c_s}(x,t_s)$$
<!-- LaTeX original:
A(c_s \to c_t) = \sum_{x \in \mathcal{D}} \sum_{t_s, t_t} \frac{\partial
 a^{l_t}_{c_t}(x,t_t)}{\partial a^{l_s}_{c_s}(x,t_s)} \cdot a^{l_s}_{c_s}(x,t_s) \cdot
 g^{l_s}_{c_s}(x,t_s)
-->


 where the sum over $(t_s, t_t)$ ranges over $t_s = t_t$ for position-wise layers and $t_s \leq t_t$
 for key/value-to-output edges in attention. In practice, we compute this sum over the training
 dataset using a distributed pipeline across multiple GPUs. To make attributions comparable across
 components with different activation scales and different frequencies of causal importance, we
 normalize by the total causal importance of the source and the root-mean-square activation of the
 target:


<label id="eq:normalized_attr"/>
$$\hat{A}(c_s \to c_t) = \frac{A(c_s \to c_t)}{\left(\sum_{x,t} g^{l_s}_{c_s}(x,t)\right) \cdot
 \text{RMS}(a^{l_t}_{c_t})}$$
<!-- LaTeX original:
\label{eq:normalized_attr}
 \hat{A}(c_s \to c_t) = \frac{A(c_s \to c_t)}{\left(\sum_{x,t} g^{l_s}_{c_s}(x,t)\right) \cdot
 \text{RMS}(a^{l_t}_{c_t})}
-->

 where $\text{RMS}(a^{l_t}_{c_t}) = \sqrt{\frac{1}{|\mathcal{D}|} \sum_{x,t} (a^{l_t}_{c_t}(x,t))^2}$
 and $|\mathcal{D}|$ is the total number of tokens processed. Dividing by the source's cumulative
 causal importance puts the attribution on a per-occurrence scale (analogous to averaging over only
 the datapoints where the source is active), while dividing by the target's RMS activation accounts
 for the target's overall magnitude. Together, these normalizations allow meaningful comparison of
 attribution strengths across edges in the graph.

 We also compute an absolute-value variant $A_{\text{abs}}(c_s \to c_t)$, which replaces the target
 activation $a^{l_t}_{c_t}$ with its absolute value $|a^{l_t}_{c_t}|$ in the backward pass. This
 variant captures the total magnitude of influence irrespective of sign, and is useful for identifying
 strong interactions where the signed attribution may cancel across datapoints.



**Prompt-level attributions.**

 For analyzing individual prompts, we compute position-aware attributions without aggregation. Given a
 prompt and a set of "alive" subcomponents (those with nonzero causal importance at each position),
 we compute the gradient attribution for each pair of alive source and target subcomponents at each
 valid combination of source and target positions. The resulting position-aware graph enables detailed
 analysis of how the model processes a specific input.

 The main changes are: separate $t_s$ and $t_t$ indices throughout, an explicit paragraph explaining
 when they differ (K/V to O edges within an attention block), and the dataset sum now ranges over
 valid position pairs rather than a single shared position.

### Post-hoc causal importance optimization


 <label id="app:posthoc_ci"/>
During VPD base training, the causal importance function $\Gamma$ is trained to predict which subcomponents are necessary to reconstruct the target model's *full output distribution across all sequence positions*. However, when analyzing a specific behavior—such as the model's prediction of a particular token at a particular position—many causally important subcomponents will be irrelevant to that specific behavior, even though they are necessary for reconstructing the full output. To isolate only the subcomponents involved in a behavior of interest, we optimize new causal importance values *post hoc* on a single prompt, using a reconstruction loss that targets only the specific aspect of the output we wish to study.

#### Setup

 Given a trained VPD model and a prompt $x$, we first run the model's trained causal importance
 function to obtain the base causal importance values $g^l_c(x,t)$ for all subcomponents on that
 prompt. We then identify the set of *alive* subcomponents for the prompt: those for which $g^l_c(x,t) > 0$ at any position $t$. Only alive subcomponents are eligible for inclusion in the post-hoc optimization, though masks for the other components can still be sampled stochastically to ensure they remain ablatable. We parameterize the new causal importances using pre-sigmoid parameters $\phi^l_c(t)$, one per alive subcomponent at each position. The causal importance values are obtained by passing these parameters through the same lower-leaky hard sigmoid function $\sigma$ (see Appendix <ref>app:sigmoids</ref>) used during base training:


$$\tilde{g}^l_c(x,t) = \sigma(\phi^l_c(t))$$
<!-- LaTeX original:
\tilde{g}^l_c(x,t) = \sigma(\phi^l_c(t))
-->


 The parameters $\phi^l_c(t)$ are initialized to the pre-sigmoid values produced by the base causal
 importance function on this prompt, providing a warm start. Non-alive subcomponents have their causal
 importance fixed at zero throughout optimization.


#### Loss function


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


The importance minimality loss $\mathcal{L}_{\text{importance-minimality}}$ has the same form as in base training (Equations <ref>eq:minimal</ref> and <ref>eq:freq_minimality</ref>), applied to the post-hoc causal importances $\tilde{g}^l_c(x,t)$. This loss encourages the optimization to find the sparsest set of subcomponents that can still reconstruct the targeted behavior. The coefficient $\lambda_{\text{min}}$ controls the sparsity–fidelity trade-off: larger values yield sparser graphs with fewer active subcomponents, potentially at the cost of reconstruction quality.

#### Masking during optimization


As in base training, the post-hoc causal importances define masks on the subcomponents via:

$$m^l_c(x,t,r) = \tilde{g}^l_c(x,t) + (1 - \tilde{g}^l_c(x,t)) r^l_c(x,t)$$
<!-- LaTeX original:
m^l_c(x,t,r) = \tilde{g}^l_c(x,t) + (1 - \tilde{g}^l_c(x,t)) r^l_c(x,t)
-->


where $r^l_c(x,t) \in [0,1]$. On each optimization step, we sample masks by drawing $r^l_c(x,t)$, either stochastically uniformly or adversarially, and compute the reconstruction loss under those masks. This ensures that the optimization satisfies the same mechanistic faithfulness criterion as base training: subcomponents marked as unimportant must be ablatable in any combination without affecting the targeted output.

$$\mathcal{L}_{\text{post-hoc}} = \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{recon}}
+\lambda_{\text{min}} \cdot \mathcal{L}_{\text{importance-minimality}}
+ \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{adversarial-recon}}$$
<!-- LaTeX original:
\mathcal{L}_{\text{post-hoc}} = \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{recon}}
+\lambda_{\text{min}} \cdot \mathcal{L}_{\text{importance-minimality}}
+ \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{adversarial-recon}}
-->


where $\mathcal{L}_{\text{adversarial-recon}}$ is computed similarly to Equation <ref>eq:adv_recon</ref>, but using the post-hoc causal importances and the targeted reconstruction loss. There is also one additional constraint imposed on the adversarial sampler compared to VPD base training: Only alive subcomponents on the prompt have their masks adversarially optimised, other subcomponents have their masks drawn stochastically. This is because we want to prevent the adversary from finetuning on data dependent noise inside the many inactive components of the model, see Section <ref>sec:methods_adv</ref>. In base training, this is accomplished by forcing the adversary to use the same $r^l_c$ for many data points. For post-hoc optimisation we cannot do this, because we only have a single prompt available. But the causal importance function has already pre-filtered the components to exclude those that were not involved in computing the prompt at all, so we attempt to sidestep this issue by restricting the adversary to components that were alive on the original prompt.

#### Optimization procedure


We optimize the pre-sigmoid parameters $\phi^l_c(t)$ using AdamW with a cosine learning rate schedule and brief linear warmup. The model weights and subcomponent parameters ($U$, $V$) are frozen throughout; only the post-hoc causal importance parameters are updated. The optimization typically converges within a few hundred steps, since it starts from a good initialization and optimizes over a single prompt rather than a dataset. The result is a set of refined causal importance values $\tilde{g}^l_c(x,t)$ that are sparser than the base values: many subcomponents that were causally important for the full output are driven to zero importance when only a specific behavior is targeted. The surviving subcomponents—those with $\tilde{g}^l_c(x,t) > 0$—form the nodes of the interaction graph for that behavior, and gradient attributions (Section <ref>app:gradient_attributions</ref>) are then computed between them.

## Training Details and Hyperparameters

<label id="app:training-details"/>

**Target model training.**

The target model architecture is described in Section <ref>sec:langauge-model-details</ref> and Table <ref>model-hyperparams</ref>.
It was trained on a subset of The Pile <cite>gao2020pile</cite> for $100,000$ steps with batch size $1024$ and context length $512$.
We used Adam <cite>kingma2017adam</cite> with learning rate $3 \times 10^{-4}$ (cosine decay to $10\%$), weight decay $0.1$, gradient clipping at $1.0$, and $600$ warmup steps.
Training used `bfloat16` mixed precision and `torch.compile`.

**VPD training.**

VPD decomposes 24 weight matrices (6 per layer: `c_fc`, `down_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`) into rank-one subcomponents with delta components enabled.
Training ran for $400,000$ steps with batch size $64$ on the same Pile dataset with context length $512$.
The component and CI function parameters were jointly optimized with AdamW (weight decay $0$), initial learning rate $5 \times 10^{-5}$ with cosine decay to $10\%$ of the initial value.
Component gradients were clipped at norm $0.01$.
One stochastic mask sample ($S=1$) was drawn per step.
Faithfulness warmup ran for $400$ steps (AdamW, lr $= 10^{-3}$, weight decay $0$), optimizing only the component parameters against $\mathcal{L}_{\mathrm{faithfulness}}$ before the main training loop.
The output divergence measure $D$ is KL divergence throughout.

The causal importance function $\Gamma$ is a shared bidirectional transformer (architecture described in Table <ref>tab:ci-hyperparams</ref>).
It takes RMS-normalized concatenations of all 24 pre-weight activations (total input dimension $D = 27,648$) and outputs $C_{\mathrm{total}} = 39,936$ causal importance values via a leaky hard sigmoid ($\alpha = 0.01$).

The $p$-norm exponent in both $\mathcal{L}_{\mathrm{importance\text{-}minimality}}$ and $\mathcal{L}_{\mathrm{frequency\text{-}minimality}}$ is linearly annealed from $p_0 = 2.0$ to $p_{\mathrm{final}} = 0.4$ over the full training run (see Section <ref>app:methods</ref> for details).

**Adversarial reconstruction.**

To optimizes the persistent sources in the persistent PGD adversarial loss, an Adam optimizer was used with $\beta_1 = 0.5$, $\beta_2 = 0.99$, learning rate $0.01$ (constant schedule with $2.5\%$ warmup).
Sources are scoped per batch element per sequence position (i.e. each individual batch element and sequence position has its own source), and each source receives $2$ warmup PGD steps per training step before the final loss computation.
Stochastic and adversarial reconstruction losses both use uniform-$k$-subset routing, where a random subset of the 24 weight matrices is masked on each step.

**Loss terms and coefficients.**

Table <ref>tab:vpd-loss-coefficients</ref> lists all loss terms and their coefficients.

<label id="tab:vpd-loss-coefficients"/>
| **Loss term** | **Reference** | **Coefficient** |
|---|---|---|
| $\mathcal{L}_{\mathrm{faithfulness}}$ (component–weight MSE) | Eq.[TODO: ?] | $10^{7}$ |
| $\mathcal{L}_{\mathrm{stochastic\text{-}recon\text{-}subset}}$ (stochastic KL) | Eq. <ref>eq:random_recon</ref> | $0.5$ |
| $\mathcal{L}_{\mathrm{adversarial\text{-}recon\text{-}subset}}$ (persistent PGD KL) | Eq. <ref>eq:adv_recon</ref> | $0.5$ |
| $\mathcal{L}_{\mathrm{importance\text{-}minimality}}$ ($\ell_p$ on CI values) | Eq. <ref>eq:minimal</ref> | $2 \times 10^{-4}$ |
*VPD loss terms and their coefficients. The importance minimality loss uses $p$-annealing from $2.0$ to $0.4$ with $\beta = 0.5$ (see text). All reconstruction losses use KL divergence.*

**Subcomponent counts.**

Table <ref>tab:vpd-subcomponent-counts</ref> lists the number of rank-one subcomponents $C$ we give to each module at initialization.

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

The CI function architecture is shown in Table <ref>tab:ci-hyperparams</ref>.

The training and evaluation losses achieved by the primary training run studied in this paper are listed in Table <ref>tab:vpd-eval-losses</ref> and Table <ref>tab:vpd-train-losses</ref> respectively.

<label id="tab:vpd-train-losses"/>
| Loss | Value |
|---|---|
| Total | $24.62$ |
| FaithfulnessLoss | $0.00000240$ |
| StochasticReconSubsetLoss | $0.2419$ |
| PersistentPGDReconLoss | $0.5733$ |
| ImportanceMinimalityLoss | $1102.0$ |
*Training losses (Measured at final step).*

<label id="tab:vpd-eval-losses"/>
| Loss | Value |
|---|---|
| StochasticReconSubsetLoss | $0.2381$ |
| PGDReconLoss | $0.9268$ |
| StochasticHiddenActsReconLoss | $0.4130$ |
| CIHiddenActsReconLoss | $0.8464$ |
*Evaluation reconstruction losses.*

## Decomposition summary statistics and properties

<label id="app:decomp-stats"/>

### VPD recovers an acceptable amount of training compute compared with other methods

Our parameter components recover an acceptable amount of the training compute spent to train the target model. If we exclude the $\Delta$ component (which is trained to be as causally unimportant as possible), the remaining parameter components, when unmasked, recover about $85\%$ of the pretraining compute. When using stochastic masks, this drops to around $30\%$. This metric is important, but unfortunately rarely reported, so comparisons to other methods are difficult. Nonetheless, VPD compares favourably to the only other method in the literature that we are aware of that reports this metric: Top-$k$ SAEs <cite>gao2024scalingevaluatingsparseautoencoders</cite> reports a pretraining loss recovered of $10\%$ when replacing a single layer of GPT-4 with an SAE with 16 million latents. Our whole model decomposition, by contrast, recovers between $11\%$ and $30\%$ while replacing the entire model, not just a single layer.

| **Masking mode** (excluding $\Delta$-component) | **CE loss** | **Training compute recovered** (%) |
|---|---|---|
| Unmasked *(All masks$=$1)* | $2.72$ | $85.2$ |
| Stochastic masks | $2.84$ | $29.8$ |
| CIs used as masks | $2.99$ | $11.4$ |
| Rounded masks *(Mask$=$1 if CI$>$0)* | $2.94$ | $13.6$ |
| **Target model** | $2.71$ | $100.0$ |
*SPD decomposition quality by masking mode.*

The adversarial loss greatly improves the SPD model performance for small source ($r$) values (Figure <ref>fig:adv-vs-no-adv</ref>).

<figure>
<label id="app:fig:adv_vs_no_adv"/><label id="fig:adv-vs-no-adv"/>
<img src="figures/adv_vs_no_adv.png">
<figcaption>Comparison between a decomposition with and without adversarial loss. The training configuration is otherwise identical. The CE loss is especially improved for small values of $r$.</figcaption>
</figure>

| Layer | $C$ | Alive | Mean L0 | L0 / $C$ (%) |
|---|---|---|---|---|
| Layer 0 | $9728$ | $9682$ | $44.10$ | $0.5$ |
| Layer 1 | $9728$ | $9663$ | $18.78$ | $0.2$ |
| Layer 2 | $9728$ | $9664$ | $51.40$ | $0.5$ |
| Layer 3 | $9728$ | $9672$ | $91.68$ | $0.9$ |
| Total | $38912$ | $38681$ | $206.0$ | $0.5$ |
*Sparsity: component counts and CI-L0 per layer.*
