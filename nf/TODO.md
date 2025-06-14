- [ ] Configure strength of prior flow with learnable temperature param.
    (This is kinda also controlled by the flexibility of the flow.)
- [x] Investigate proper mu and sigma prior scale values.
- [ ] AI Checker
- [ ] Gumbel Model
- [ ] Model Clarity Writing
- [ ] CNF Background
    - [ ] Make adjoint clearer.
    - [ ] Synopsis of what CNF gives us.
    - [ ] Grad outputs writing.
- [x] Investigate narrower hidden layers and shallower networks:
    - [x] Weaker/strong priors.
    - [x] Better posterior models.
- [x] Not KNN adjacency. Use distance instead.
- [x] Allow mclust initialization.
- [ ] Stack a CNF/MAF together?
- [ ] Attempt a CNF with wider hidden units instead of deeper?
- [ ] Way more data, but unempirical prior.
- [x] GIN and GAT implementations.
- [ ] Convert color schemes to rainbow.
- [ ] MoNeT
- [ ] Change mean and scale LR for un-empirical case.


class ClampedCNF(zuko.flows.continuous.CNF):
    def forward(self, x, context=None):
        output, logdet = super().forward(x, context)
        return output.clamp(min=-10.0, max=10.0), logdet  # Adjust min/max as needed

# Then use it like:
cluster_probs_flow_dist = ClampedCNF(
    features=num_clusters,
    context=context_length,
    hidden_features=hidden_layers,
    activation=retrieve_activation(activation),
    exact=False,
)