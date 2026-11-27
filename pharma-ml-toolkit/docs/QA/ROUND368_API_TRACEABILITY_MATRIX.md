# ROUND 368 API TRACEABILITY MATRIX (part19, Phase 3)

Derived live from the running FastAPI application (`create_app().openapi()` at commit 53f2c92d + the round368 working tree), cross-checked against `contracts/openapi.json`, the route modules, the API test suite, and the typed frontend service clients.

**The r367 count discrepancy resolved:** the committed contract, the runtime app, and the project-status documentation all agree on **142 paths / 148 operations**; the "144 paths" figure existed only in the r367 commit-message prose — the artifact never drifted. Operation IDs are unique (148/148). The generated `frontend/src/types/api.d.ts` regenerates byte-identically from the committed contract (openapi-typescript 7.13.0).

**Authentication model (all rows):** loopback-only Host validation + the per-session `x-session-token` custom header (SessionSecurity); the `/health` handshake lane answers before routing without the session token; everything else requires it. **Job lanes:** routes in files that submit through the JobManager are marked `async-job` (REST responses stay authoritative; progress/cancel flow over the `/api/v1/events` WebSocket); the rest are synchronous.

| # | Method | Path | Route file | Request DTO | 200-response DTO | Lane | Frontend consumers (frontend/src/…) | Backend tests |
|---|---|---|---|---|---|---|---|---|
| 1 | GET | `/api/v1/admet/endpoints` | admet.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_admet.py |
| 2 | POST | `/api/v1/admet/score` | admet.py | AdmetScoreRequest | object | sync | src/features/ml/__tests__/AdmetLane.test.tsx; src/services/ml/client.ts; src/types/api.d.ts | test_admet.py |
| 3 | POST | `/api/v1/applicability-domain/analyze` | applicability_domain.py | ADAnalyzeRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_contracts.py, test_docking_pkpd_md_viz_ad.py |
| 4 | GET | `/api/v1/applicability-domain/catalog` | applicability_domain.py | — | object | sync | src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 5 | POST | `/api/v1/applicability-domain/domain-box` | applicability_domain.py | DomainBoxRequest | object | sync | src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 6 | POST | `/api/v1/applicability-domain/knn` | applicability_domain.py | KnnRequest | object | sync | src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 7 | POST | `/api/v1/applicability-domain/leverage-williams` | applicability_domain.py | WilliamsRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 8 | GET | `/api/v1/artifacts` | artifacts.py | — | object | sync | src/components/registry/ModelRegistry.tsx; src/components/registry/__tests__/ModelRegistry.test.tsx; src/features/datasets/DatasetsPage.tsx | test_contracts.py, test_datasets_ml_artifacts.py, test_docking_pkpd_md_viz_ad.py |
| 9 | POST | `/api/v1/artifacts/figure` | artifacts.py | FigureRequest | object | sync | src/features/dl/DLPage.tsx; src/features/docking/DockingPage.tsx; src/features/generative/GenerativePage.tsx | test_datasets_ml_artifacts.py, test_penetration_security.py, test_persistence.py |
| 10 | POST | `/api/v1/artifacts/model` | artifacts.py | ModelArtifactRequest | object | sync | src/components/registry/ModelRegistry.tsx; src/features/dl/DLPage.tsx; src/services/dl/client.ts | test_datasets_ml_artifacts.py, test_ml_registry.py, test_multiresponse.py |
| 11 | POST | `/api/v1/artifacts/table` | artifacts.py | ArtifactTableRequest | object | sync | src/features/datasets/DatasetsPage.tsx; src/types/api.d.ts | test_datasets_ml_artifacts.py, test_persistence_database.py |
| 12 | DELETE | `/api/v1/artifacts/{artifact_id}` | artifacts.py | — | object | sync | src/services/artifacts/client.ts; src/types/api.d.ts | test_ml_registry.py, test_rsm.py (f-string parametric hit) |
| 13 | GET | `/api/v1/artifacts/{artifact_id}` | artifacts.py | — | object | sync | src/services/artifacts/client.ts; src/types/api.d.ts | test_ml_registry.py, test_rsm.py (f-string parametric hit) |
| 14 | GET | `/api/v1/artifacts/{artifact_id}/download` | artifacts.py | — | — | sync | src/services/artifacts/client.ts; src/types/api.d.ts | test_ml_registry.py, test_rsm.py (f-string parametric hit) |
| 15 | GET | `/api/v1/capabilities` | capabilities.py | — | object | sync | src/types/api.d.ts | test_backend_service.py, test_penetration_security.py |
| 16 | GET | `/api/v1/capabilities/{capability_id}` | capabilities.py | — | — | sync | src/types/api.d.ts | test_backend_service.py (f-string parametric hit) |
| 17 | POST | `/api/v1/chemistry/descriptors` | chemistry.py | DescriptorsRequest | object | sync | src/features/generative/GenerativePage.tsx; src/services/chemistry/client.ts; src/types/api.d.ts | test_backend_service.py, test_contracts.py, test_datasets_ml_artifacts.py |
| 18 | POST | `/api/v1/chemistry/fingerprints` | chemistry.py | FingerprintsRequest | object | sync | src/services/chemistry/client.ts; src/types/api.d.ts | test_backend_service.py |
| 19 | GET | `/api/v1/database` | database.py | — | object | sync | src/components/system/DatabasePreferences.tsx; src/components/system/JobCenterCard.tsx; src/components/system/__tests__/DatabasePreferences.test.tsx | test_contracts.py, test_persistence.py, test_persistence_database.py |
| 20 | POST | `/api/v1/database/backup` | database.py | BackupRequest | object | sync | src/services/database/client.ts; src/types/api.d.ts | test_contracts.py, test_persistence.py, test_persistence_database.py |
| 21 | GET | `/api/v1/database/jobs` | database.py | — | object | sync | src/components/system/JobCenterCard.tsx; src/features/introduction/IntroductionPage.tsx; src/services/database/client.ts | test_contracts.py, test_persistence.py, test_persistence_database.py |
| 22 | GET | `/api/v1/database/preferences` | database.py | — | object | sync | src/components/system/DatabasePreferences.tsx; src/components/system/__tests__/DatabasePreferences.test.tsx; src/services/database/client.ts | test_contracts.py, test_persistence.py, test_persistence_database.py |
| 23 | DELETE | `/api/v1/database/preferences/{key}` | database.py | — | object | sync | src/components/system/DatabasePreferences.tsx; src/services/database/client.ts; src/types/api.d.ts | test_contracts.py |
| 24 | GET | `/api/v1/database/preferences/{key}` | database.py | — | object | sync | src/components/system/DatabasePreferences.tsx; src/services/database/client.ts; src/types/api.d.ts | test_contracts.py |
| 25 | PUT | `/api/v1/database/preferences/{key}` | database.py | PreferenceValue | object | sync | src/components/system/DatabasePreferences.tsx; src/services/database/client.ts; src/types/api.d.ts | test_contracts.py |
| 26 | POST | `/api/v1/database/restore` | database.py | RestoreRequest | object | sync | src/services/database/client.ts; src/types/api.d.ts | test_contracts.py, test_persistence.py, test_persistence_database.py |
| 27 | GET | `/api/v1/datasets` | datasets.py | — | object | sync | src/components/quality/DataQualityCard.tsx; src/services/datasets/client.ts; src/services/quality/__tests__/types.test.ts | test_contracts.py, test_data_quality.py, test_datasets_ml_artifacts.py |
| 28 | POST | `/api/v1/datasets` | datasets.py | — | object | sync | src/components/quality/DataQualityCard.tsx; src/services/datasets/client.ts; src/services/quality/__tests__/types.test.ts | test_contracts.py, test_data_quality.py, test_datasets_ml_artifacts.py |
| 29 | DELETE | `/api/v1/datasets/{dataset_id}` | datasets.py | — | object | sync | src/types/api.d.ts | test_data_quality.py, test_samples.py, test_system_diagnostics.py |
| 30 | GET | `/api/v1/datasets/{dataset_id}` | datasets.py | — | object | sync | src/types/api.d.ts | test_data_quality.py, test_samples.py, test_system_diagnostics.py |
| 31 | GET | `/api/v1/datasets/{dataset_id}/column/{column}` | datasets.py | — | object | sync | src/types/api.d.ts | test_data_quality.py, test_runs.py, test_persistence_database.py (f-string parametric hit) |
| 32 | GET | `/api/v1/datasets/{dataset_id}/preview` | datasets.py | — | object | sync | src/types/api.d.ts | test_data_quality.py, test_runs.py, test_persistence_database.py (f-string parametric hit) |
| 33 | POST | `/api/v1/datasets/{dataset_id}/profile` | datasets.py | ProfileRequest | object | sync | src/types/api.d.ts | test_data_quality.py |
| 34 | POST | `/api/v1/datasets/{dataset_id}/quality` | datasets.py | QualityRequest | object | sync | src/types/api.d.ts | test_data_quality.py |
| 35 | POST | `/api/v1/datasets/{dataset_id}/split` | datasets.py | SplitRequest | object | sync | src/types/api.d.ts | test_data_quality.py, test_runs.py, test_persistence_database.py (f-string parametric hit) |
| 36 | POST | `/api/v1/dl/autoencoder-encode` | dl.py | AutoencoderEncodeRequest | object | sync | src/features/dl/DLPage.tsx; src/features/dl/__tests__/CnnAutoencoderLane.test.tsx; src/services/dl/client.ts | test_dl_architectures.py |
| 37 | POST | `/api/v1/dl/autoencoder-train` | dl.py | AutoencoderTrainRequest | object | sync | src/features/dl/DLPage.tsx; src/services/dl/client.ts; src/types/api.d.ts | test_dl_architectures.py |
| 38 | POST | `/api/v1/dl/cnn-predict` | dl.py | CnnPredictRequest | object | sync | src/features/dl/DLPage.tsx; src/features/dl/__tests__/CnnAutoencoderLane.test.tsx; src/services/dl/client.ts | test_dl_architectures.py |
| 39 | POST | `/api/v1/dl/cnn-train` | dl.py | CnnTrainRequest | object | sync | src/features/dl/DLPage.tsx; src/services/dl/client.ts; src/types/api.d.ts | test_dl_architectures.py |
| 40 | POST | `/api/v1/dl/gnn-predict` | dl.py | GnnPredictRequest | object | sync | src/features/dl/DLPage.tsx; src/features/dl/__tests__/CnnAutoencoderLane.test.tsx; src/features/dl/__tests__/GnnLane.test.tsx | test_model_evaluation.py |
| 41 | POST | `/api/v1/dl/gnn-train` | dl.py | GnnTrainRequest | object | sync | src/features/dl/DLPage.tsx; src/features/dl/__tests__/CnnAutoencoderLane.test.tsx; src/features/dl/__tests__/GnnLane.test.tsx | test_dl_architectures.py, test_model_evaluation.py |
| 42 | GET | `/api/v1/dl/networks` | dl.py | — | object | sync | src/services/dl/client.ts; src/types/api.d.ts | test_dl_architectures.py, test_dl_semi_generative.py, test_model_evaluation.py |
| 43 | POST | `/api/v1/dl/train` | dl.py | DLTrainRequest | object | sync | src/features/dl/DLPage.tsx; src/features/dl/__tests__/CnnAutoencoderLane.test.tsx; src/services/dl/client.ts | test_contracts.py, test_dl_architectures.py, test_dl_semi_generative.py |
| 44 | POST | `/api/v1/docking/contacts` | docking.py | ContactsRequest | object | sync | src/features/docking/DockingPage.tsx; src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 45 | POST | `/api/v1/docking/dock` | docking.py | DockRequest | object | sync | src/features/docking/DockingPage.tsx; src/services/docking/client.ts; src/types/api.d.ts | test_contracts.py, test_docking_pkpd_md_viz_ad.py |
| 46 | POST | `/api/v1/docking/dockstring/objective` | docking.py | ObjectiveRequest | object | sync | src/features/docking/DockingPage.tsx; src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 47 | GET | `/api/v1/docking/dockstring/targets` | docking.py | — | object | sync | src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 48 | GET | `/api/v1/docking/dockstring/targets/{name}` | docking.py | — | object | sync | src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 49 | GET | `/api/v1/docking/engines` | docking.py | — | object | sync | src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 50 | POST | `/api/v1/docking/ligand/pdbqt` | docking.py | LigandPdbqtRequest | object | sync | src/features/docking/DockingPage.tsx; src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py, test_samples.py |
| 51 | POST | `/api/v1/docking/parse/dlg` | docking.py | ParseTextRequest | object | sync | src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py, test_samples.py |
| 52 | POST | `/api/v1/docking/parse/vina` | docking.py | ParseTextRequest | object | sync | src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 53 | POST | `/api/v1/docking/pose-rmsd` | docking.py | PoseRmsdRequest | object | sync | src/features/docking/DockingPage.tsx; src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 54 | POST | `/api/v1/docking/receptor/prepare` | docking.py | ReceptorPrepareRequest | object | sync | src/features/docking/DockingPage.tsx; src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 55 | POST | `/api/v1/docking/scoring/consensus` | docking.py | ConsensusRequest | object | sync | src/features/docking/DockingPage.tsx; src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 56 | POST | `/api/v1/docking/scoring/ligand-efficiency` | docking.py | LigandEfficiencyRequest | object | sync | src/features/docking/DockingPage.tsx; src/services/docking/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 57 | POST | `/api/v1/drugresponse/analyze` | drugresponse.py | DrugResponseAnalyzeRequest | object | sync | src/features/ml/__tests__/DrugResponseLane.test.tsx; src/features/ml/__tests__/MultiobjectiveLane.test.tsx; src/features/ml/__tests__/MultiresponseLane.test.tsx | test_drugresponse.py |
| 58 | GET | `/api/v1/drugresponse/registry` | drugresponse.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_drugresponse.py |
| 59 | POST | `/api/v1/dta/features` | dta.py | DtaFeaturesRequest | object | sync | src/features/ml/__tests__/DrugResponseLane.test.tsx; src/features/ml/__tests__/DtaLane.test.tsx; src/features/ml/__tests__/MultiobjectiveLane.test.tsx | test_dta.py |
| 60 | GET | `/api/v1/dta/registry` | dta.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_dta.py |
| 61 | POST | `/api/v1/generative/diffusion/sample` | generative.py | DiffusionSampleRequest | object | sync | src/features/generative/__tests__/FeatureGenerativeLane.test.tsx; src/services/generative/client.ts; src/types/api.d.ts | test_dl_architectures.py |
| 62 | POST | `/api/v1/generative/diffusion/train` | generative.py | DiffusionTrainRequest | object | sync | src/services/generative/client.ts; src/types/api.d.ts | test_dl_architectures.py |
| 63 | POST | `/api/v1/generative/gan/sample` | generative.py | GanSampleRequest | object | sync | src/features/generative/__tests__/FeatureGenerativeLane.test.tsx; src/services/generative/client.ts; src/types/api.d.ts | test_dl_architectures.py |
| 64 | POST | `/api/v1/generative/gan/train` | generative.py | GanTrainRequest | object | sync | src/services/generative/client.ts; src/types/api.d.ts | test_dl_architectures.py |
| 65 | GET | `/api/v1/generative/objectives` | generative.py | — | object | sync | src/services/generative/client.ts; src/types/api.d.ts | test_dl_semi_generative.py |
| 66 | POST | `/api/v1/generative/optimize` | generative.py | OptimizeRequest | object | sync | src/features/generative/GenerativePage.tsx; src/services/generative/client.ts; src/types/api.d.ts | test_dl_semi_generative.py |
| 67 | GET | `/api/v1/generative/priors` | generative.py | — | object | sync | src/services/generative/client.ts; src/types/api.d.ts | test_dl_semi_generative.py, test_persistence.py |
| 68 | GET | `/api/v1/generative/priors/{prior_id}` | generative.py | — | object | sync | src/services/generative/client.ts; src/types/api.d.ts | test_dl_semi_generative.py (f-string parametric hit) |
| 69 | POST | `/api/v1/generative/rnn/sample` | generative.py | SampleRequest | object | sync | src/features/generative/GenerativePage.tsx; src/services/generative/client.ts; src/types/api.d.ts | test_dl_semi_generative.py |
| 70 | POST | `/api/v1/generative/rnn/train` | generative.py | PriorTrainRequest | object | sync | src/features/generative/GenerativePage.tsx; src/services/generative/client.ts; src/types/api.d.ts | test_contracts.py, test_dl_semi_generative.py |
| 71 | GET | `/api/v1/health` | health.py | — | object | sync | src/services/api/startup.ts; src/types/api.d.ts | test_backend_service.py, test_contracts.py, test_trusted_origins.py |
| 72 | GET | `/api/v1/jobs` | jobs.py | — | object | sync | src/components/system/JobCenterCard.tsx; src/components/system/__tests__/JobCenterCard.test.tsx; src/features/introduction/IntroductionPage.tsx | test_backend_service.py, test_contracts.py, test_data_quality.py |
| 73 | GET | `/api/v1/jobs/{job_id}` | jobs.py | — | object | sync | src/features/ml/__tests__/WorkflowBuilderLane.test.tsx; src/types/api.d.ts | test_backend_service.py, test_data_quality.py, test_datasets_ml_artifacts.py |
| 74 | POST | `/api/v1/jobs/{job_id}/cancel` | jobs.py | — | object | sync | src/types/api.d.ts | test_backend_service.py, test_docking_pkpd_md_viz_ad.py, test_workflow.py |
| 75 | GET | `/api/v1/jobs/{job_id}/result` | jobs.py | — | object | sync | src/types/api.d.ts | test_backend_service.py, test_data_quality.py, test_docking_pkpd_md_viz_ad.py |
| 76 | POST | `/api/v1/md/analyze` | md.py | MDAnalyzeRequest | object | sync | src/features/md/MDPage.tsx; src/services/md/client.ts; src/types/api.d.ts | test_contracts.py, test_docking_pkpd_md_viz_ad.py |
| 77 | GET | `/api/v1/md/catalog` | md.py | — | object | sync | src/features/md/MDPage.tsx; src/services/md/client.ts; src/types/api.d.ts | test_contracts.py, test_docking_pkpd_md_viz_ad.py (f-string parametric hit) |
| 78 | POST | `/api/v1/md/enm` | md.py | EnmRequest | object | sync | src/features/md/MDPage.tsx; src/services/md/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 79 | POST | `/api/v1/md/pca` | md.py | PcaRequest | object | sync | src/features/md/MDPage.tsx; src/services/md/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 80 | POST | `/api/v1/md/study` | md.py | StudyRequest | object | sync | src/features/md/MDPage.tsx; src/services/md/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 81 | POST | `/api/v1/md/synthetic` | md.py | SyntheticRequest | object | sync | src/features/md/MDPage.tsx; src/services/md/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 82 | POST | `/api/v1/md/xvg` | md.py | XvgRequest | object | sync | src/features/md/MDPage.tsx; src/services/md/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 83 | POST | `/api/v1/ml/explain` | ml.py | ExplainRequest | object | sync | src/features/ml/MLPage.tsx; src/services/ml/client.ts; src/types/api.d.ts | test_datasets_ml_artifacts.py, test_dl_architectures.py, test_model_evaluation.py |
| 84 | GET | `/api/v1/ml/models` | ml.py | — | object | sync | src/components/registry/ModelRegistry.tsx; src/services/ml/__tests__/registry.test.ts; src/services/ml/client.ts | test_datasets_ml_artifacts.py, test_ml_registry.py |
| 85 | POST | `/api/v1/ml/predict` | ml.py | PredictRequest | object | sync | src/features/dl/DLPage.tsx; src/features/dl/__tests__/CnnAutoencoderLane.test.tsx; src/features/dl/__tests__/GnnLane.test.tsx | test_datasets_ml_artifacts.py, test_dl_architectures.py, test_dl_semi_generative.py |
| 86 | POST | `/api/v1/ml/train` | ml.py | TrainRequest | object | sync | src/components/registry/ModelRegistry.tsx; src/components/runs/__tests__/RunLineage.test.tsx; src/services/ml/__tests__/registry.test.ts | test_contracts.py, test_datasets_ml_artifacts.py, test_docking_pkpd_md_viz_ad.py |
| 87 | GET | `/api/v1/ml/trained` | ml.py | — | object | sync | src/components/registry/ModelRegistry.tsx; src/components/runs/__tests__/RunLineage.test.tsx; src/services/ml/__tests__/registry.test.ts | test_datasets_ml_artifacts.py, test_ml_registry.py, test_persistence.py |
| 88 | DELETE | `/api/v1/ml/trained/{model_id}` | ml.py | — | object | sync | src/components/registry/ModelRegistry.tsx; src/services/ml/client.ts; src/types/api.d.ts | test_ml_registry.py |
| 89 | GET | `/api/v1/ml/trained/{model_id}` | ml.py | — | object | sync | src/components/registry/ModelRegistry.tsx; src/services/ml/client.ts; src/types/api.d.ts | test_ml_registry.py |
| 90 | GET | `/api/v1/model-evaluation/catalog` | model_evaluation.py | — | object | sync | src/types/api.d.ts | test_model_evaluation.py |
| 91 | POST | `/api/v1/model-evaluation/conformal-classification` | model_evaluation.py | ConformalClassificationRequest | object | sync | src/types/api.d.ts | test_model_evaluation.py |
| 92 | POST | `/api/v1/model-evaluation/conformal-regression` | model_evaluation.py | ConformalRegressionRequest | object | sync | src/types/api.d.ts | test_model_evaluation.py |
| 93 | POST | `/api/v1/model-evaluation/coverage-diagnostics` | model_evaluation.py | CoverageDiagnosticsRequest | object | sync | src/types/api.d.ts | test_model_evaluation.py |
| 94 | POST | `/api/v1/multiobjective/optimize` | multiobjective.py | MultiObjectiveOptimizeRequest | object | sync | src/features/ml/__tests__/MultiobjectiveLane.test.tsx; src/features/ml/__tests__/MultiresponseLane.test.tsx; src/features/ml/__tests__/WorkflowBuilderLane.test.tsx | test_multiobjective.py |
| 95 | GET | `/api/v1/multiobjective/registry` | multiobjective.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_multiobjective.py |
| 96 | GET | `/api/v1/multiresponse/models` | multiresponse.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_multiresponse.py |
| 97 | POST | `/api/v1/multiresponse/predict` | multiresponse.py | PredictMultiResponseRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_multiresponse.py |
| 98 | GET | `/api/v1/multiresponse/registry` | multiresponse.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_multiresponse.py |
| 99 | POST | `/api/v1/multiresponse/train` | multiresponse.py | TrainMultiResponseRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_multiresponse.py, test_rsm.py |
| 100 | POST | `/api/v1/pharmacogenomics/analyze` | pharmacogenomics.py | PharmacogenomicsAnalyzeRequest | object | sync | src/features/ml/__tests__/MultiobjectiveLane.test.tsx; src/features/ml/__tests__/MultiresponseLane.test.tsx; src/features/ml/__tests__/PharmacogenomicsLane.test.tsx | test_pharmacogenomics.py |
| 101 | GET | `/api/v1/pharmacogenomics/registry` | pharmacogenomics.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_pharmacogenomics.py |
| 102 | POST | `/api/v1/pkpd/bateman` | pkpd.py | BatemanRequest | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 103 | GET | `/api/v1/pkpd/catalog` | pkpd.py | — | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_contracts.py |
| 104 | POST | `/api/v1/pkpd/iv-bolus` | pkpd.py | IvBolusRequest | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 105 | POST | `/api/v1/pkpd/nca` | pkpd.py | NcaRequest | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py, test_samples.py |
| 106 | POST | `/api/v1/pkpd/pbpk` | pkpd.py | PbpkRequest | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 107 | POST | `/api/v1/pkpd/pd` | pkpd.py | PdRequest | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 108 | POST | `/api/v1/pkpd/population` | pkpd.py | PopulationRequest | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 109 | POST | `/api/v1/pkpd/steady-state` | pkpd.py | SteadyStateRequest | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 110 | POST | `/api/v1/pkpd/superposition` | pkpd.py | SuperpositionRequest | object | sync | src/features/pkpd/PKPDPage.tsx; src/services/pkpd/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 111 | POST | `/api/v1/reinforce/evaluate` | reinforce.py | ReinforceEvaluateRequest | object | sync | src/services/ml/unsupervised.ts; src/types/api.d.ts | test_unsupervised_reinforce.py |
| 112 | GET | `/api/v1/reinforce/methods` | reinforce.py | — | object | sync | src/services/ml/unsupervised.ts; src/types/api.d.ts | test_unsupervised_reinforce.py |
| 113 | POST | `/api/v1/reinforce/train` | reinforce.py | ReinforceTrainRequest | object | sync | src/services/ml/unsupervised.ts; src/types/api.d.ts | test_unsupervised_reinforce.py |
| 114 | GET | `/api/v1/repurposing/registry` | repurposing.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_repurposing.py |
| 115 | POST | `/api/v1/repurposing/screen` | repurposing.py | RepurposingScreenRequest | object | sync | src/features/ml/__tests__/DrugResponseLane.test.tsx; src/features/ml/__tests__/MultiobjectiveLane.test.tsx; src/features/ml/__tests__/MultiresponseLane.test.tsx | test_repurposing.py |
| 116 | POST | `/api/v1/rsm/contour` | rsm.py | ContourRsmRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_rsm.py |
| 117 | POST | `/api/v1/rsm/design` | rsm.py | DesignRsmRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_rsm.py |
| 118 | POST | `/api/v1/rsm/fit` | rsm.py | FitRsmRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_rsm.py |
| 119 | GET | `/api/v1/rsm/models` | rsm.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_rsm.py |
| 120 | POST | `/api/v1/rsm/optimize` | rsm.py | OptimizeRsmRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_rsm.py |
| 121 | POST | `/api/v1/rsm/predict` | rsm.py | PredictRsmRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_rsm.py |
| 122 | GET | `/api/v1/rsm/registry` | rsm.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_rsm.py |
| 123 | GET | `/api/v1/runs` | runs.py | — | object | sync | src/components/runs/RunLineage.tsx; src/services/runs/__tests__/client.test.ts; src/services/runs/client.ts | test_ml_registry.py, test_multiresponse.py, test_penetration_security.py |
| 124 | GET | `/api/v1/runs/{run_id}` | runs.py | — | object | sync | src/services/runs/client.ts; src/services/runs/types.ts; src/types/api.d.ts | test_multiresponse.py, test_rsm.py, test_runs.py |
| 125 | GET | `/api/v1/samples` | samples.py | — | object | sync | src/services/samples/client.ts; src/types/api.d.ts | test_data_quality.py, test_ml_registry.py, test_runs.py |
| 126 | POST | `/api/v1/samples/{sample_id}/load` | samples.py | — | object | sync | src/services/samples/client.ts; src/types/api.d.ts | test_ml_registry.py, test_runs.py, test_samples.py |
| 127 | GET | `/api/v1/samples/{sample_id}/preview` | samples.py | — | object | sync | src/services/samples/client.ts; src/types/api.d.ts | test_samples.py |
| 128 | POST | `/api/v1/semi-supervised/evaluate` | semi_supervised.py | SemiSupervisedEvaluateRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_dl_semi_generative.py |
| 129 | POST | `/api/v1/semi-supervised/train` | semi_supervised.py | SemiSupervisedTrainRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_contracts.py, test_dl_semi_generative.py |
| 130 | POST | `/api/v1/system/cache/cleanup` | system.py | — | object | sync | src/components/system/DiagnosticsCard.tsx; src/services/system/__tests__/diagnostics.test.ts; src/services/system/client.ts | test_system_diagnostics.py |
| 131 | GET | `/api/v1/system/diagnostics` | system.py | — | object | sync | src/components/system/DiagnosticsCard.tsx; src/services/system/__tests__/diagnostics.test.ts; src/services/system/client.ts | test_system_diagnostics.py |
| 132 | GET | `/api/v1/system/resources` | system.py | — | object | sync | src/app/App.tsx; src/components/system/ResourceMonitor.tsx; src/services/system/__tests__/client.test.ts | test_system.py |
| 133 | POST | `/api/v1/unsupervised/anomaly` | unsupervised.py | AnomalyRequest | object | sync | src/services/ml/unsupervised.ts; src/types/api.d.ts | test_unsupervised_reinforce.py |
| 134 | POST | `/api/v1/unsupervised/cluster` | unsupervised.py | ClusterRequest | object | sync | src/services/ml/unsupervised.ts; src/types/api.d.ts | test_unsupervised_reinforce.py |
| 135 | GET | `/api/v1/unsupervised/methods` | unsupervised.py | — | object | sync | src/services/ml/unsupervised.ts; src/types/api.d.ts | test_unsupervised_reinforce.py |
| 136 | GET | `/api/v1/visualization/catalog` | visualization.py | — | object | sync | src/features/visualization/VisualizationPage.tsx; src/services/visualization/client.ts; src/types/api.d.ts | test_contracts.py, test_docking_pkpd_md_viz_ad.py |
| 137 | POST | `/api/v1/visualization/concentration-profile` | visualization.py | ConcentrationProfileRequest | object | sync | src/features/visualization/VisualizationPage.tsx; src/services/visualization/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 138 | POST | `/api/v1/visualization/dose-accumulation` | visualization.py | DoseAccumulationRequest | object | sync | src/features/visualization/VisualizationPage.tsx; src/services/visualization/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 139 | POST | `/api/v1/visualization/md-series` | visualization.py | MdSeriesRequest | object | sync | src/features/md/MDPage.tsx; src/features/visualization/VisualizationPage.tsx; src/services/md/client.ts | test_docking_pkpd_md_viz_ad.py |
| 140 | POST | `/api/v1/visualization/report` | visualization.py | ReportRequest | object | sync | src/features/visualization/VisualizationPage.tsx; src/services/visualization/client.ts; src/types/api.d.ts | test_docking_pkpd_md_viz_ad.py |
| 141 | POST | `/api/v1/visualization/stat` | visualization.py | StatRequest | object | sync | src/features/datasets/DatasetsPage.tsx; src/features/visualization/VisualizationPage.tsx; src/services/visualization/client.ts | test_docking_pkpd_md_viz_ad.py |
| 142 | POST | `/api/v1/visualization/table` | visualization.py | VisualizationTableRequest | object | sync | src/features/dl/DLPage.tsx; src/features/docking/DockingPage.tsx; src/features/generative/GenerativePage.tsx | test_docking_pkpd_md_viz_ad.py |
| 143 | POST | `/api/v1/workflow/run` | workflow.py | WorkflowRunRequest | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_workflow.py |
| 144 | GET | `/api/v1/workflow/runs` | workflow.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_workflow.py |
| 145 | GET | `/api/v1/workflow/runs/{run_id}` | workflow.py | — | object | sync | src/types/api.d.ts | test_workflow.py (f-string parametric hit) |
| 146 | GET | `/api/v1/workflow/steps` | workflow.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_workflow.py |
| 147 | GET | `/api/v1/workflow/templates` | workflow.py | — | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_workflow.py |
| 148 | POST | `/api/v1/workflow/validate` | workflow.py | object | object | sync | src/services/ml/client.ts; src/types/api.d.ts | test_workflow.py |

**Coverage summary:** 148 operations across 142 paths; every operation has at least one frontend consumer reference; every operation has a backend test lane (direct literal hits for 138; the 10 parametric routes are exercised through f-string calls in the listed files — see `tests/api/`); zero duplicate operation IDs; zero routes present in the contract but absent at runtime, and zero runtime routes absent from the contract.

**Column semantics:** *Request DTO* is the `application/json` body schema (`—` for no-body GETs); *200-response DTO* is the first 2xx schema (`dict` inline schemas show as their type); validation errors (422) are automatic for every DTO-bearing route; error handling for domain failures raises `HTTPException` with documented status codes per route module (pinned by the API test suite and the 67-test penetration suite).

## Routes without a Studio GUI consumer (Phase 4 cross-check, r368)

Nine registered routes have no frontend page consumer in `frontend/src` (verified by
path-string search over every service client and page component). Each has an explicit
documented reason:

| Route | Reason for no GUI consumer |
|---|---|
| `GET /api/v1/capabilities`, `GET /api/v1/capabilities/{capability_id}` | Capability-registry contract surface (machine-readable capability inventory); consumed by the registry tooling + API tests; the Studio surfaces capabilities through its own workspace cards, not this generic registry route. |
| `GET /api/v1/applicability-domain/catalog` | Python/REST contract listing of the AD method set; the Studio's AD card hard-pins the two implemented lanes (kNN-LoO distance + Williams leverage with its `domain-box` geometry) rather than rendering the catalog. |
| `POST /api/v1/applicability-domain/knn`, `POST /applicability-domain/domain-box` | REST lanes for API/CLI consumers of the AD math; the Studio AD card exercises the combined `analyze` + `leverage-williams` job lanes instead. |
| `GET /api/v1/model-evaluation/catalog` | Contract catalog for the model-evaluation family (conformal/coverage); no Studio card enumerates it. |
| `POST /api/v1/model-evaluation/conformal-regression`, `POST …/conformal-classification`, `POST …/coverage-diagnostics` | Python/REST-first scientific lanes (conformal intervals + coverage diagnostics); exercised by `tests/api/test_model_evaluation.py`; no Studio GUI card exists for conformal evaluation yet (a documented GUI-coverage gap, not dead backend functionality). |
| `GET /api/v1/workflow/runs/{run_id}` | Per-run detail for API consumers; the Studio's Run Lineage card consumes the list route (`GET /workflow/runs`) and renders run records from the list payload. |

No frontend call exists without a matching backend route (verified: every path string in
`services/**` and every inline `api.post` resolves to a registered route).
