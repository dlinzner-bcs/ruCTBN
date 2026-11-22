# ruCTBN

Rust implementation of continuous-time Bayesian networks (CTBNs), with structure learning, trajectory sampling, exact state-space dynamics, the SSL-CTBN node-star forward/backward approximation, and an optional disjoint-cluster mean-field extension.

## Build and test

```bash
cargo build
cargo test
cargo run
```

The package uses the stable Rust toolchain declared in `rust-toolchain`.

## Dynamics inference

The library exposes three numerical propagation methods:

- `exact_dynamics`: propagates the full joint CTBN master equation.
- `star_variational_dynamics`: convenience entry point for a fixed CIM. It runs the SSL-CTBN node-star fixed point with a degenerate rate model, where the arithmetic and geometric posterior expected rates coincide.
- `star_variational_dynamics_with_rates`: the paper-faithful entry point for uncertain or mixture posterior CIMs. It accepts separate arithmetic and geometric expected rate matrices and maintains one singleton forward marginal `mu_i`, one backward response `rho_i`, child-response coupling, and expected dwell/transition statistics per node. Parent configurations are averaged from products of singleton parent marginals; no disjoint cluster partition is supplied.
- `cluster_mean_field_dynamics`: an optional disjoint-cluster forward extension. It retains joint correlations inside user-selected clusters and factorizes between clusters. Its old name, `cluster_variational_dynamics`, remains as a compatibility alias, but manually defined disjoint clusters are not the clustering used by the MATLAB/Python SSL-CTBN node-star method.

All propagators use fourth-order Runge-Kutta integration. The star implementation follows Equations 10–12 and Algorithm 1 of the SSL-CTBN paper, including noisy-observation jump conditions.

```rust
use ctbn::dynamics::{cluster_mean_field_dynamics, exact_dynamics};
use ctbn::star::{
    star_variational_dynamics_with_rates, StarConfig, StarRateModel,
};

// Three-node coupled v-structure: X0 -> X1 <- X2.
let exact = exact_dynamics(&ctbn, &[0, 0, 0], 1.0, 0.002)?;
let rate_model = StarRateModel::degenerate(&ctbn);
let star = star_variational_dynamics_with_rates(
    &ctbn,
    &rate_model, // In learning, supply distinct R and tilde-R matrices here.
    &[0, 0, 0],
    &[],
    1.0,
    0.002,
    &StarConfig::default(),
)?;
let cluster_extension = cluster_mean_field_dynamics(
    &ctbn,
    &[0, 0, 0],
    &[vec![0], vec![1], vec![2]],
    1.0,
    0.002,
)?;
```

```mermaid
flowchart LR
    X0[Node 0 state X₀]
    X1[Node 1 state X₁]
    X2[Node 2 state X₂]
    X0 --> X1
    X2 --> X1

    subgraph Star[Node-star SSL approximation]
        C0((μ₀, ρ₀)) -. factorized parent belief .-> C1((μ₁, ρ₁))
        C2((μ₂, ρ₂)) -. factorized parent belief .-> C1
    end

    subgraph Exact[Exact inference]
        J[Joint state X₀, X₁, X₂]
    end
```

### Node-star equations

For node `i`, the star solver alternates the equations corresponding to the historical MATLAB implementation and StarCTBN reference:

1. Compute the parent-configuration distribution as a product of singleton parent beliefs:

   \[
   q_i^u(t) \approx \prod_{j\in\operatorname{pa}(i)} q_j(t,u_j).
   \]

2. Compute two parent-averaged matrices. The arithmetic posterior expectation `R_i` is used in diagonal backward dynamics and in the subtraction term of the child response. The geometric posterior mean `tilde R_i` is used for off-diagonal backward/forward flow and expected transition counts. The distinction is essential: for mixture/posterior CIMs, generally `R_i != tilde R_i`. Only the degenerate fixed-CIM case satisfies equality, explaining why that special no-evidence fixed point reduces to singleton mean field.

   \[
   \Psi_i(x,t)=\sum_c\sum_{z\neq z'} q_c(z,t)q_c^{u\setminus i}(t)
   \left(\tilde R_c(z,z'\mid x,u\setminus i)
   \frac{\rho_c(z',t)}{\rho_c(z,t)}-R_c(z,z'\mid x,u\setminus i)\right).
   \]

3. Integrate `rho_i` backward between observations using geometric off-diagonal rates and the arithmetic diagonal exit rate, applying each observation likelihood as a jump.
4. Integrate `mu_i` forward with tilted geometric rates `tilde R_i(x,x') rho_i(x') / rho_i(x)`.
5. Integrate dwell statistics from `mu_i` and parent marginals, and transition statistics with the geometric rate and `rho_i(x') / rho_i(x)` ratio.

The implementation exposes the resulting `dwell_times` and `transitions` in `StarResult` for the EM maximization step.

This is why the method uses overlapping **node-parent/child stars**, not a manual disjoint partition. The compatibility cluster API is a separate approximation and should not be confused with it.

Exact inference stores every global state, so its state space grows as the product of local dimensions. The optional disjoint-cluster extension stores only the state product for each configured cluster.

```mermaid
flowchart TD
    Model[CTBN conditional intensity matrices]
    Model --> Exact
    Model --> Star
    Model --> Cluster
    Model --> Sampling

    Exact[Full joint master equation]
    Star[Singleton mu/rho node stars]
    Cluster[Optional disjoint joint clusters]
    Sampling[Gillespie trajectories]

    Exact --> Compare{Compare marginals and evidence}
    Star --> Compare
    Cluster --> Compare
    Sampling --> Compare
```

## Validation

The dynamics tests validate:

1. Exact propagation against the analytic two-state solution.
2. Exact marginals against 20,000 seeded Gillespie paths.
3. Node-star marginals and expected statistics against exact single-node dynamics.
4. Noisy terminal observations against the reweighted posterior probability.
5. Distinct arithmetic/geometric rate matrices to ensure node-star uses geometric expected rates for forward flow rather than silently collapsing to plain mean field.
6. Optional disjoint-cluster mean-field marginals against exact inference.
7. Probability normalization and non-negativity throughout propagation.

Run only the dynamics checks with:

```bash
cargo test dynamics::tests
```

The faithful star solver follows the parent-neighborhood equations in the historical `ssl-ctbn` MATLAB code and the cleaner [StarCTBN Python reference](https://github.com/AdrianSosic/StarCTBN). Parent neighborhoods define the conditional rates, but are not stored as joint clusters: their configurations are averaged using products of singleton marginals. “Subsets” in the MATLAB learning code are sparse-regression rate components, not variational clusters. The Rust package does not depend on either historical implementation at runtime.

## Benchmarks

Run the reproducible release benchmark with:

```bash
cargo run --release --example dynamics_benchmark
```

Workload: three binary nodes in the coupled v-structure `X0 -> X1 <- X2`, duration `0.7`, integration step `0.002`, 100 inference repetitions, and 20,000 seeded sampled paths. The collider `X1` has rates conditional on both parents. Measurements are machine-dependent and exclude Cargo compilation time.

### Reference results

Measured on macOS 13.6, Intel Core i7-7820HQ at 2.90 GHz, Rust stable:

| Method | Average time | Collider `P(state=1)` |
|---|---:|---:|
| Exact master equation | 9.427 ms/run | 0.475448 |
| Singleton disjoint mean-field extension | 9.280 ms/run | 0.518534 |
| Gillespie sampling | 4.176 µs/path | 0.474150 |

The singleton disjoint-mean-field maximum absolute marginal error in this workload is `0.043086`. The six-node comparison below exercises a non-degenerate node-star model with separate `R` and `tilde R`, demonstrating that the paper-specific rate handling is active rather than silently collapsing to singleton mean field.

```mermaid
xychart-beta
    title "Average runtime (milliseconds)"
    x-axis ["Exact", "Cluster variational", "Sampling path"]
    y-axis "Runtime (ms)" 0 --> 12
    bar [9.427, 9.280, 0.004176]
```

```mermaid
xychart-beta
    title "Collider final probability"
    x-axis ["Exact", "Cluster variational", "Sampling"]
    y-axis "P(state=1)" 0 --> 0.6
    bar [0.475448, 0.518534, 0.474150]
```

In this three-node v-structure, exact inference tracks eight joint states while the singleton-cluster solver tracks six one-node states. The collider's conditional rates depend on both parents, illustrating the interaction between coupled parents. Larger networks increase the full joint state space exponentially; increasing cluster size trades additional runtime and memory for retained correlations and exactness.

### Six-node solver comparison

The six-node benchmark includes two coupled v-structures and a directed cycle. Nodes 0 and 1 feed node 2; nodes 3, 4, and 2 feed node 5; and node 5 feeds node 0, forming `X0 -> X2 -> X5 -> X0`. This gives the cluster solver two directed dependencies crossing between its two three-node clusters.

```mermaid
flowchart LR
    X0[Node 0] --> X2[Node 2]
    X1[Node 1] --> X2
    X3[Node 3] --> X5[Node 5]
    X4[Node 4] --> X5
    X2 --> X5
    X5 --> X0

    subgraph A[Cluster A]
        X0
        X1
        X2
    end

    subgraph B[Cluster B]
        X3
        X4
        X5
    end
```

The benchmark compares exact inference, the faithful SSL-CTBN node-star solver with non-degenerate arithmetic/geometric expected rates, plain singleton mean field, the optional two three-node cluster extension, and direct Gillespie sampling. The six-node workload uses 20 inference repetitions and 20,000 sampled paths. The node-star benchmark is therefore a different posterior-rate model from the base fixed-CIM methods, rather than another spelling of singleton mean field.

| Solver | Average time | Node 5 `P(state=1)` | Maximum marginal error |
|---|---:|---:|---:|
| Exact master equation | 138.558 ms/run | 0.432803 | 0 |
| SSL node-star with distinct `R`/`tilde R` | 144.943 ms/run | 0.406795 | 0.026008 |
| Singleton mean field | 25.627 ms/run | 0.471635 | 0.040265 |
| Two 3-node clusters | 49.650 ms/run | 0.435751 | 0.002948 |
| Gillespie sampling | 11.374 µs/path | 0.435600 | — |

```mermaid
xychart-beta
    title "Six-node solver runtime (milliseconds)"
    x-axis ["Exact", "Node-star", "Singleton", "3-node clusters", "Sampling path"]
    y-axis "Runtime (ms)" 0 --> 160
    bar [138.558, 144.943, 25.627, 49.650, 0.011374]
```

```mermaid
xychart-beta
    title "Six-node node 5 final probability"
    x-axis ["Exact", "Node-star", "Singleton", "3-node clusters", "Sampling"]
    y-axis "P(state=1)" 0 --> 0.55
    bar [0.432803, 0.406795, 0.471635, 0.435751, 0.435600]
```

Here, exact inference has 64 joint states. The faithful node-star solver tracks 12 singleton state values plus 12 backward-response values, while the separate disjoint extension tracks 16 cluster-state values when configured as two three-node clusters. The benchmark gives the collider nodes distinct arithmetic and geometric posterior means (`tilde R` off-diagonals are 12–18% below `R`), so the paper-specific rate handling is exercised and node-star no longer collapses to singleton mean field. For a deterministic CIM or degenerate posterior mixture, `R = tilde R`; then the no-observation node-star fixed point does reduce to singleton mean field. The directed cycle is valid because CTBNs use cyclic directed interaction graphs, not a static Bayesian-network DAG.

### Initial and final state

The benchmark starts every node in state 0. At `t = 0`, therefore, `P(X_i = 1) = 0` for all six nodes. The final values compare the full solver with the two three-node cluster solver:

| Node | Initial `P(state=1)` | Exact final `P(state=1)` | Cluster final `P(state=1)` |
|---|---:|---:|---:|
| 0 | 0.000000 | 0.377888 | 0.378430 |
| 1 | 0.000000 | 0.263691 | 0.263691 |
| 2 | 0.000000 | 0.417626 | 0.419725 |
| 3 | 0.000000 | 0.282976 | 0.282976 |
| 4 | 0.000000 | 0.477564 | 0.477564 |
| 5 | 0.000000 | 0.432803 | 0.435751 |

```mermaid
xychart-beta
    title "Node marginals: initial versus final"
    x-axis ["Node 0", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
    y-axis "P(state=1)" 0 --> 0.6
    bar [0, 0, 0, 0, 0, 0]
    bar [0.377888, 0.263691, 0.417626, 0.282976, 0.477564, 0.432803]
    bar [0.378430, 0.263691, 0.419725, 0.282976, 0.477564, 0.435751]
```

In chart order, the series are initial state, exact final marginals, and clustered final marginals. The initial series remains on the baseline, while the exact and clustered final bars overlap closely where the approximation is accurate.

### Fixed-state evidence benchmark

For endpoint evidence, both the full initial state and a full final state are fixed. This evaluates the probability of observing the exact joint-state transition, not merely independent node marginals:

```text
X(0) = [0, 0, 0, 0, 0, 0]
X(T) = [0, 1, 0, 1, 0, 1]
```

`DynamicsResult::final_state_probability` reads the matching joint state for exact inference. For cluster approximations, it multiplies the matching state probabilities of all disjoint cluster beliefs.

| Solver | Endpoint evidence probability |
|---|---:|
| Exact | 0.003922 |
| SSL node-star with distinct `R`/`tilde R` | 0.006149 |
| Singleton mean field | 0.006119 |
| Two 3-node clusters | 0.005385 |
| Gillespie sampling | 0.003700 |

The node-star evidence uses the explicit arithmetic/geometric rate model from the benchmark, while exact, cluster mean field, and sampling use the base fixed CIM. Its result therefore is not expected to coincide with the fixed-CIM solvers. The sampling estimate uses 20,000 independent seeded paths and counts only paths whose complete final state matches the evidence. This event is relatively rare, so its Monte Carlo standard error is approximately `sqrt(p(1-p)/20000)`, about `0.00043` for this workload. Larger endpoint samples improve the estimate, while additional cluster size retains more cross-node correlation in the approximated evidence value.

```mermaid
xychart-beta
    title "Fixed endpoint evidence probability"
    x-axis ["Exact", "Node-star", "Singleton", "3-node clusters", "Sampling"]
    y-axis "Probability" 0 --> 0.007
    bar [0.003922, 0.006149, 0.006119, 0.005385, 0.003700]
```

To add a benchmark to the tables, update `examples/dynamics_benchmark.rs`, run it in release mode several times, and report the machine, workload, and full result rather than relying on debug-build timings.
