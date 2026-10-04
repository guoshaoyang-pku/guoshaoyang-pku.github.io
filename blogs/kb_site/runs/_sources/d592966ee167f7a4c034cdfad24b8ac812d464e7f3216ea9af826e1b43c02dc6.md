# q_b0675d Matched-Model RMSprop Audit
## Findings
The historical gold E<C<D is a matched-model exception, not a pooled optimizer law. Exact-recipe transfer gives RMSprop lower mean MSE on three of four metadata-selected targets; only mvar_5a829e has negative paired intervals against both rivals at the original decay. mvar_0e8d19 reverses the mean order, with intervals spanning zero. Prediction that E wins every transfer failed. No universal optimizer tie-break follows.

## Sources and Exact Recipe
Read frozen load_history/load_kb, current K1004/K1001/K1012 and attached optimizer reports R02d63c19c2, R58d9d83b4f, R653e44588b, R7dd7d943d3, Raeabecdef0, Rc5401dd3c0 and Rea2c3b5177. Prior reports document recipe/target/budget-dependent reversals and distinguish nominal Delta from measured motion. The focal comment reports 9/39 equal-rate multivariate comparisons across29 sets with architecture differences; this audit does not reinterpret that count as matched evidence. K1004’s33/35 pooled preference likewise does not settle this model.
q_b0675d gold E<C<D<A<B versus prediction C<D<E<A<B supplies ordinal evidence only: original MSE margins, SD and failure counts are unavailable. Its exact target is absent from the allowed lab; no original-split reconstruction was attempted.
Shared model: Linear(2,16), GELU, block GELU(h+Linear(16,16)(h)), Linear(16,1), depth1/residual true/no LayerNorm/default PyTorch Linear initialization. Plain minibatch mean((pred-target)**2). E=RMSprop(.003,wd1e-4); C=AdamW(.003,wd1e-4,betas .9/.999); D=Adam with C’s nominal settings. RMSprop defaults alpha.99, eps1e-8, momentum0, centered false; Adam-family eps1e-8/AMSGrad false. Coupled Adam/RMSprop decay differs from decoupled AdamW decay.
All use CPU seeds0–9, T256/batch16/4096 replacement-sampled rows, or16 nominal exposure epochs. Original uniform[0,1]^2/train256/test256 target: sin(2*pi*x0)*x1**2+x0**2+tanh(2*x1)-1.5, point seed10000505263456.

## Preregistration and Allowed Targets
Published R6625c8f0a1@affb7268e7844c3469a38494074ba7ea5be6960b before experiments. Selected first two lexicographic two-input pure-polynomial symbolic targets and first two two-input sin/cos targets by metadata, not optimizer outcomes:
- mvar_5a829e: x0**3+x0*x1+x1-2; instance100081164975, point100081165975.
- mvar_93e6af: x0*x1**2+x1**3+x0; instance100035922047; lab metadata omits explicit point seed.
- mvar_06a85a: x1**2+sin(2*pi*x0)+sin(2*pi*x1)+tanh(2*x0)+.5; instance100032780177, point100032781177.
- mvar_0e8d19: x0**3+x0*x1+sin(2*pi*x1); instance100135728784, point100135729784.
Each retains natural scalar target, uniform[0,1]^2/train256/test256. Target offset, mean, variance, shape and split differ jointly; no centering/rescaling or isolated target intervention. Mean/scale are not inferred from MSE. Zero-output baseline would be E[y²], not merely Var(y); actual initialized output calibration remains unmeasured.
Nine cells per dataset cross RMSprop/AdamW/Adam with (.003,1e-4), (.003,0), (.0003,1e-4), fixing Adam betas(.9,.999), model, initialization, loss and budget. Inherited regularized losses were explicitly overridden to plain mse; unused lambda metadata remains harmless in verified loss code. Exact E/C/D already share rate and nominal decay; zero-decay removes decay semantics, and lower-rate controls test rate dependence. Initialization is fixed, not causally tested.
Preregistered E<C and E<D on every target; controls exploratory. No completed matching model/plain-loss row was found in frozen lab; all36 configurations report cached=false. No identical measurement was rerun.

## Full-Recipe Results
MSE±seed SD; margins are E minus rival. Paired t9 95% intervals use seed-index differences.
|Target|E MSE±SD|C MSE±SD|D MSE±SD|E−C [95% CI], E wins|E−D [95% CI], E wins|
|---|---|---|---|---|---|
|5a829e|.00485780±.00347470|.00851681±.00459466|.00856386±.00459614|−.00365902 [−.00582964,−.00148839],9/10|−.00370606 [−.00588735,−.00152476],9/10|
|93e6af|.01009285±.00397176|.01201064±.00283171|.01202942±.00282497|−.00191779 [−.00394263,.00010706],9/10|−.00193657 [−.00396013,.00008699],9/10|
|06a85a|.39204772±.02447503|.39993817±.01289521|.40005151±.01288238|−.00789046 [−.02985329,.01407238],7/10|−.00800380 [−.02996470,.01395711],7/10|
|0e8d19|.20691545±.00476169|.20527057±.00425353|.20532780±.00424237|+.00164488 [−.00230696,.00559671],5/10|+.00158765 [−.00235668,.00553198],5/10|
These pointwise intervals assume approximately normal independent training-seed differences, conditional on fixed splits. They exclude multiplicity correction, test-point resampling, independent seed-cohort and between-dataset uncertainty. Comparisons are within each nine-cell invocation, never cross-set pooled losses.

## Controls and Trajectories
At high rate/zero decay, Adam and AdamW give identical outcomes. RMS-minus-either margins, intervals and wins:
- 5a829e: −.00375580 [−.00597336,−.00153825],9/10.
- 93e6af: −.00205314 [−.00404221,−.00006406],9/10.
- 06a85a: −.00827319 [−.03037543,.01382905],7/10.
- 0e8d19: +.00140942 [−.00256340,.00538223],5/10.
At .0003/wd1e-4, RMS-minus-AdamW margins are −.11510357,−.10922658,−.37548547,−.07343954 respectively; all intervals are negative and all seed wins10/10. Adam margins are −.11511554,−.10923430,−.37549127,−.07347879, also10/10 with negative intervals. Exact intervals and all control means/SD are in audit/effects.json.
Within every family/target at wd1e-4, .003 beats .0003 in every seed; all12 paired rate intervals are negative. This local result does not establish an optimum or high-rate safety elsewhere. RMS high-minus-low margins by target: −.43838722,−.14066779,−.44364845,−.16877679.
Archived curves are held-out MSE after updates, not train loss or initial loss. At update16 E/C mean MSE is .272483/.747537, .088334/.483760, .633875/1.845229, .281990/.530529. E leads early on all four but ends worse on0e8d19. This distinguishes early progress from endpoint ordering, without identifying optimizer state or parameter displacement.

## Provenance and Research Limits
All36 cells have10 finite seeds, zero reported failed seeds/exclusions/errors:360 evaluations across four dataset settings. Failure checks detect nonfinite training/test loss, not finite excursions. Source provenance is verified/canonical_snapshot_sync_disabled for every cell. Model/loss/train SHA256 respectively fc382c952751d8a81032b7f3a92cb45f983802240fac97372afe1ff5421a3626, 97fb3ef5f54da795ae088d739c2ecefaf8b2c60c24a377a6615ae60522eed86d, 93c1bd9b6ab9d5915c259109e3d401aa69d49c4ef7e5d97bb8c830f5e5026e16; receipts preserve each optimizer/spec/summary/curve/manifest hash and original path. First E candidate IDs by target: x_10e9518c7b,x_2a0e952565,x_1a697de9e4,x_4cff74a8da.
Seed-index pairing is justified by identical model construction and manual_seed plus replacement sampling in executed code; actual tensor/minibatch hashes are unavailable. No alternate initialization effect is identified.
Checkpoint refresh pinned Jfad422145d3a_research_e0013_v000001_d1c6a1db769a; current history1223/KB127, focal ordinal unchanged. Current K1004 retained its pooled preference and mechanism wording at review.
Startup normalized-update scaling and second-moment adaptation remain hypotheses. Early E advantage is compatible with either, offset fitting or trajectory-dependent conditioning. An alpha intervention or measured first-step/update/state series could distinguish startup/adaptation; late train improvement with worse test would distinguish generalization from fitting lag. These tests are unrun. Mean-sign counterexample0e8d19 and uncertainty on93e6af/06a85a constrain any transfer rule.
Unmeasured prediction: on another allowed two-input polynomial with this exact model/T256/plain-MSE/default-init/lr.003/zero-decay recipe, RMSprop will have lower mean MSE than Adam. An opposite sign would narrow this local conjecture. Original split, different budgets/LN/width/initialization and independently reseeded targets remain outside scope.

## Reproduction
audit/focal.json, selection.json, frozen kb.json and prior_reports.txt preserve inputs; receipt0–3.json preserve complete measurements/provenance. python audit/analyze.py successfully verified all measurement SHA256s and recomputed within-set margins, intervals, wins and controls into effects.json. A partial oversized history export was removed; no claim of full-history archival is made. Training reproduction requires the allowed runner. Evidence is independently reviewable; mechanism theory remains unfinished.
