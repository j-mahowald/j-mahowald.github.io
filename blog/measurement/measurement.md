# Measuring in machine learning

Here, I aim to give an overview of how mathematicians study measurement rigorously, and how machine-learning research can use the rigor of the mathematical measure to its advantage.

As a refresher, there are two separate – yet highly related and annoyingly conflatable – concepts to keep track of here: a *metric* and a *measure*. I'll start with the metric and get into the measure later.

## The metric

### Mathematical basis
A *metric* on a set $S$ is a tool that gives a sense of *distance between two points in a set*. It is a function $d: S \times S \to \mathbb{R}$ that follows three rules:
- **Non-negativity**: $d(x,y) \geq 0$ for any pair of $x$ and $y$ in $S$. The only way we can have $d(x,y)=0$ is if $x=y$. That is, any two points that are not identical have some distance between them.
- **Symmetry**: $d(x,y)=d(y,x)$. That is, your distance starting from point $x$ and walking to point $y$ is the same as the distance starting from point $y$ and walking to point $x$.
- **Triangle inequality**: for any three points $x$, $y$, and $z$ in $S$, $d(x,z) \leq d(x,y) + d(y,z)$. This one's a bit subtler, but it generalizes the fact that no edge of a triangle can be longer than the sum of the other two edges. Usually, when we're determining whether a given function is a metric, this is the condition we need to look out for.

The simplest example of a metric is the absolute value $|\cdot|$ on the real numbers: $|x-y|$ is always non-negative (by definition), it's definitely symmetric ($|x-y| = |-(y-x)| = |y-x|$), and it passes the triangle inequality.

### Metrics in machine learning

Metrics are important because they define most of the topologies we work with in machine learning. When we say a metric $d$ "defines" a topology on a set $S$, we mean that the open sets of that topology take the form $\{x: d(s,x) < \varepsilon\}$ for a given point $s$ in $S$ and some real number $\varepsilon$. In the real numbers, for instance, the set $\{x: d(x,0) = |x-0| <1 \}$ is the same as the interval $(-1, 1)$, which is clearly an open interval.

Critically, **metrics and loss functions are not the same**. Often they don't even coincide, with the notable exception of the $L^2$ norm in regression. They serve fundamentally different purposes –– metrics define geometry, loss functions define optimization objectives –– but knowing and applying the difference is a very useful skill.

Of the metric topologies in machine learning, there are three main ones, one that shows up only sometimes, and another subtler (and very cool!) one that shows up in some interesting settings.

1. The **parameter space** $\Theta$: when defined as a Euclidean space $\Theta := \mathbb{R}^N$ for some integer $N$, the parameter space depends on the Euclidean metric between two points $x, y \in \mathbb{R}^N$ given by $$d(x,y) = ||x-y||_p = \left[\sum_{i=1}^N (x_i-y_i)^p \right]^{1/p}$$

    We tend to set $p=2$, though $p=1$ (the "Manhattan distance") and $p=\infty$ are also common in special settings. Our goal in machine learning is to move $\theta$ around in $\Theta$ such that a function $f_\theta$ parameterized by $\theta$ performs better at some task. If think of it like moving the values of a group of telescope knobs ($\theta$) around the range of possible angles ($\Theta$) so that the telescope can capture the clearest image.

2. The **data space** is simply the space that the data live in. For an image $x$ of $64 \times 64$ pixels, we can say $x$ exists in $X = \mathbb{R}^{64 \cdot 64}$.

    We can define the distance based on the space itself: in Euclidean space, we use the same metric as above, even though they're measuring entirely different things. Most times, we're not inducing movement *within* the data space, unless we're doing something funky like dataset distillation.

3. The **function space** $\mathcal{F}$: this is where $f_\theta$ itself lives. For a given set of parameters $\theta$, we can define a "realization map" $\Phi: \mathbb{R}^P \to (X \to Y)$, where $(X \to Y)$ is the set of all functions going from $X$ to $Y$, so that $\Phi(\theta) = f_\theta$. You can think of $\Phi$ as a function that "affixes" (or "realizes") a set of parameters to its functional representation. The image $\mathcal{F}=\Phi(\Theta)$ is the set of all functions that are realizable from $\Theta$.

    The most common metric on this space is the $L^p$ norm. This requires some metric on the set $Y$ given by $||\cdot ||_Y$ (you can think of this as $||y||_Y = d_Y(y, 0)$, if you like). Then, for two functions $f$ and $g$ in $\mathcal{F}$, 
    $$
    d(f,g) = ||f - g||_p = \left[\int_X ||f(x)-g(x)||^p_Y \; dx\right]^{1/p},
    $$
    where that $dx$ term is a more abstract and subtle object than calculus I would've suggested – more on that later.

4. The **space of probability distributions** $\mathcal{P}$ shows up when we're working with statistical models. Classifying an image into 10 classes, this takes the form of the 9-dimension probability simplex $\Delta_9 = \{\vec{\mu} \in \mathbb{R}^{10}: \mu_i \geq 0, \; \sum_i \mu_i = 1 \}$. A "good" probability distribution would assign most or all probability to the correct class, and training a statistical model amounts to moving our predictions around in $\mathcal{P}$ toward the good ones. 

    In general, there are a few proper metrics in this space, including the $L^p$ distance given above with $p=1$ or $2$ (if the distribution is discrete), the Hellinger distance $d(p, q) = \sqrt{\frac{1}{2} \sum_i (\sqrt{p_i} - \sqrt{q_i})^2}$, and the Fisher information metric.

5. A **tangent space** $\mathcal{T}_{p_0} (M)$ is defined at a point $p_0$ within some other space $M$ (each point in $M$ gets its own tangent space! The collection of all tangent spaces across all points in $M$ is $M$'s **tangent bundle**). An element $v \in \mathcal{T}_{p_0} (M)$ corresponds to some movement from $p_0$. There are lots of versions of tangent spaces in differential (and algebraic) geometry, but we're concerned here with two of them:
    - $\mathcal{T}_{\theta_0} (\Theta) $, the tangent space of a particular set of parameters $\theta_0$ within the space $\Theta$ of all possible parameters. 
    - $\mathcal{T}_{f_0} (\mathcal{F})$ , the tangent space of a particular function $f_0$ within the space $\mathcal{F}$ of all possible functions.

    Distance within a tangent space depends on how distance is defined in its original manifold. To unfold this relationship, we'll take a quick detour to talk about how inner products, norms, and metrics a related.

### Inner products, norms, and metrics

Most spaces that we work with in ML have an implicitly or explicitly defined **inner product**, a function that accepts two inputs and measures their alignment in the real numbers. In Euclidean space, it's the dot product: $\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x} \cdot \mathbf{y} = x_1 y_1 + \ldots + x_n y_n.$

The "alignment" interpretation here is clear: vectors pointing in the same or opposite directions have very positive or very negative inner products, respectively, while orthogonal vectors evaluate to 0. In function space, the inner product is defined analogously to the metric we discussed: for functions $f$ and $g$ defined on the same domain $D$, $\langle f, g \rangle = \int_D f(x) g(x) dx.$

We can define the norm $|| v || = \sqrt{\langle v, v \rangle}$. This is what we construct when we say an inner product "induces" a norm. This is well defined since the inner product between a vector and itself is always nonnegative: regardless of whether $x_i$ is positive or negative, $x_i x_i$ will always be nonnegative, so we can take the $\sqrt{\cdot}$ operation without fear. 

Defining distance here becomes easy: we can say $d(x,y) = ||x-y||$ for whatever norm we've defined. Indeed, our function norm $\langle f, g \rangle$ induces a norm $||f||$ which further induces the same metric $||f-g||$ we discussed above, specifically with $p=2$.

