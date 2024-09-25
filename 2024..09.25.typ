#set page(margin: 1in, numbering: "1")
#set par(leading: 0.8em, first-line-indent: 1.8em, justify: true)
#set math.equation(numbering: "(1)")
#show par: set block(spacing: 0.8em)
#show heading: set block(above: 1.4em, below: 1em)

#let argmax = $op("argmax", limits: #true)$
#let given = $thin | thin$
#let round = $op("round")$


== Signal structure

With $C in {0, 1}$ the context, $X in {0, 1, 2, 3}$ the cue, and $S in {0, 1}$ the signal, the chain structure is:

$ C -> X -> S . $

The decision-maker (DM) uses the signal structure $P( S given x )$ to inform their belief about the true context.


== Inference without history

Equipped with a prior $P( C )$ over contexts and assuming independence of contexts across trials, the DM wants to infer $C$ from $s$.
This is done according to:

$ P( C given s ) prop P( s given C ) P( C ) = paren.l sum_x P( s given x ) P( x given C ) paren.r P( C ) . $

Then, in order to maximize their probability of being correct in the trial, the DM will use the following decision rule:

$ hat(c)( s ) = argmax_c P( C = c given s ) = round paren.l P( C = 1 given s ) paren.r . $


== Inference with history

On realizing signal $s_(t - 1)$ in trial $t - 1$, the DM will update their prior over contexts using their posterior for their context in trial $t - 1$ and their belief about the transition probabilities of the context between trials (Markov hypothesis):

$ P paren.l C_t given s_(t - 1) paren.r = sum_(c_(t - 1)) P paren.l C_t given c_(t - 1) paren.r P( c_(t - 1) given s_(t - 1) ) . $

Then, this prior will be used in inference on the signal in trial $t$.
Recursively, exploiting the assumed conditional independence of $s_t$ from $s^(t - 1)$ given $c_t$:

$ P( C_t given s^t := bracket.l s_1, dots s_t bracket.r ) = P( C_t given s_t, s^(t - 1) ) prop P( s_t given C_t ) P( C_t given s^(t - 1) ) . $