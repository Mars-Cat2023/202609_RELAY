# RELAY Inference Algorithm with Gaussian Hidden-State Noise

## 1. State Initialization

For a Sudoku of length \(L=81\), define the state at inference iteration \(k\) as

\[
s_k = (x^{(k)}, h^{(k)}),
\]

where \(x^{(k)} \in \mathcal{V}^{81}\) and \(h^{(k)} \in \mathbb{R}^{81\times384}\). Initialize the token state with the given digits at clue positions and `[MASK]` at empty positions, and initialize the hidden state as

\[
h^{(0)}=0.
\]

Let \(c_i=1\) indicate an immutable clue position.

## 2. RELAY Forward Pass

At iteration \(k\), RELAY combines token embeddings with the relayed hidden state:

\[
u^{(k)} = \operatorname{Embed}_{\theta}(x^{(k)}) + \operatorname{LN}_{\mathrm{relay}}(h^{(k)}).
\]

The Transformer produces

\[
z^{(k)} = \operatorname{Transformer}_{\theta}(u^{(k)}).
\]

The same forward pass produces token logits and the Gaussian hidden-policy mean:

\[
\Lambda^{(k)} = \operatorname{LMHead}_{\theta}(\operatorname{LN}_{\mathrm{out}}(z^{(k)})),
\qquad
\mu^{(k)}=z^{(k)}.
\]

## 3. Token Probabilities

Full-vocabulary token probabilities use token temperature \(\tau_{\mathrm{tok}}>0\):

\[
P_{i,v}^{(k)} = \frac{\exp(\Lambda_{i,v}^{(k)}/\tau_{\mathrm{tok}})}{\sum_{u\in\mathcal V}\exp(\Lambda_{i,u}^{(k)}/\tau_{\mathrm{tok}})}.
\]

For every selected position \(i\), sample

\[
\widehat{x}_i^{(k)} \sim \operatorname{Categorical}(P_{i,:}^{(k)}).
\]

Equivalently, categorical sampling can use Gumbel-Max:

\[
\widehat{x}_i^{(k)} = \arg\max_v\left[\Lambda_{i,v}^{(k)}/\tau_{\mathrm{tok}} + g_{i,v}\right],
\qquad g_{i,v}\sim\operatorname{Gumbel}(0,1).
\]

## 4. Confidence-Based Position Selection

The currently masked and mutable positions are

\[
M^{(k)}=\{i:x_i^{(k)}=[\mathrm{MASK}],\ c_i=0\}.
\]

Confidence uses a separate temperature \(T_{\mathrm{conf}}\):

\[
P_{i,v}^{\mathrm{conf},(k)}=\operatorname{softmax}(\Lambda_{i,:}^{(k)}/T_{\mathrm{conf}})_v,
\qquad
a_i^{(k)}=\max_v P_{i,v}^{\mathrm{conf},(k)}.
\]

Sort mutable masked positions by decreasing confidence:

\[
a_{\pi_1}^{(k)}\geq a_{\pi_2}^{(k)}\geq\cdots\geq a_{\pi_n}^{(k)},
\qquad n=|M^{(k)}|.
\]

Define cumulative uncertainty:

\[
C_j^{(k)}=\sum_{r=1}^{j}(1-a_{\pi_r}^{(k)}).
\]

The initial selection is

\[
\widetilde U^{(k)}=\{\pi_j:C_j^{(k)}<\eta\}.
\]

The final position set is

\[
U^{(k)}=
\begin{cases}
\widetilde U^{(k)}, & \widetilde U^{(k)}\neq\varnothing,\\
\{\pi_1\}, & \widetilde U^{(k)}=\varnothing\ \text{and}\ M^{(k)}\neq\varnothing,\\
\varnothing, & M^{(k)}=\varnothing.
\end{cases}
\]

The fallback guarantees that at least one position is selected while unresolved masks remain.

## 5. Token-State Transition

Only selected positions are updated:

\[
x_i^{(k+1)}=
\begin{cases}
\widehat{x}_i^{(k)}, & i\in U^{(k)},\\
x_i^{(k)}, & i\notin U^{(k)}.
\end{cases}
\]

Clue positions are never modified. This is inference, so sampled tokens are written into the next state; there is no teacher forcing.

## 6. Gaussian Hidden-State Transition

The hidden policy is

\[
q_{\theta}(h^{(k+1)}\mid s_k)=\mathcal N(h^{(k+1)};\mu^{(k)},\sigma^2I).
\]

Sample independent Gaussian noise and update the relayed hidden state:

\[
\epsilon^{(k)}\sim\mathcal N(0,I),
\qquad
h^{(k+1)}=\mu^{(k)}+\sigma\epsilon^{(k)}=z^{(k)}+\sigma\epsilon^{(k)}.
\]

The sampled hidden state is fed into the next forward pass:

\[
u^{(k+1)}=\operatorname{Embed}_{\theta}(x^{(k+1)})+\operatorname{LN}_{\mathrm{relay}}(z^{(k)}+\sigma\epsilon^{(k)}).
\]

When \(\sigma=0\), hidden relaying is deterministic. When \(\sigma>0\), self-conditioning is stochastic.

## 7. Complete State Transition

One inference iteration is

\[
\begin{aligned}
s_k &= (x^{(k)},h^{(k)}),\\
(\Lambda^{(k)},\mu^{(k)}) &= f_{\theta}(x^{(k)},h^{(k)}),\\
U^{(k)} &= \operatorname{ConfidenceSelect}(\Lambda^{(k)},x^{(k)},c;T_{\mathrm{conf}},\eta),\\
\widehat{x}_i^{(k)} &\sim \operatorname{Categorical}(\operatorname{softmax}(\Lambda_{i,:}^{(k)}/\tau_{\mathrm{tok}})),\\
x_i^{(k+1)} &= \widehat{x}_i^{(k)}\ \text{for}\ i\in U^{(k)},\\
h^{(k+1)} &= \mu^{(k)}+\sigma\epsilon^{(k)},\\
s_{k+1} &= (x^{(k+1)},h^{(k+1)}).
\end{aligned}
\]

## 8. Termination and NFE

Inference terminates when no mutable masks remain,

\[
M^{(k+1)}=\varnothing,
\]

or when the maximum rollout length \(K_{\max}\) is reached. At the final allowed iteration, all remaining mutable mask positions are forced to be selected.

Each iteration requires one RELAY forward evaluation:

\[
\mathrm{NFE}=\text{number of executed inference iterations}.
\]

## 9. Current Experimental Configuration

\[
T_{\mathrm{conf}}=2,
\qquad
\tau_{\mathrm{tok}}=0.3,
\qquad
\eta=0.15.
\]

Two evaluation settings are used:

- \(\sigma_{\mathrm{eval}}=0\): deterministic hidden relaying.
- \(\sigma_{\mathrm{eval}}=1\): Gaussian hidden-state relaying.
