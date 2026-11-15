# Карта модулей Ziwago18 — «заявлено vs работает»

Сгенерировано из измерений: `tools/probes/module_map_probe.py` → `tools/probes/build_module_map.py`. Классификация НЕ выдумана: статус выведен из числа реальных вызовов функций в фазах обучение / чат / прямая генерация.

## Уровень 1 — файлы

| Статус | Файлов | Строк |
|---|---:|---:|
| SHIPPED | 46 | 21713 |
| PHANTOM | 4 | 1202 |
| OFF | 0 | 0 |
| DEAD | 16 | 4799 |
| **Итого** | **66** | **27714** |

| Файл | Строк | Статус | Вызовы (train/chat/gen) | Экземпляров | Роль |
|---|---:|---|---:|---:|---|
| `ziwago/action/basal_ganglia.py` | 123 | **DEAD** | 0/0/0 | — | маршрутизация действия (D1, выбора ответа) |
| `ziwago/action/contextual.py` | 396 | **SHIPPED** | 11012/2/516 | ActionGateConfig, ContextualActionGate | контекстная политика действия |
| `ziwago/app/corpora.py` | 139 | **PHANTOM** | 0/0/0 | PairedDialogueStreamer | поток диалоговых корпусов |
| `ziwago/app/curriculum.py` | 66 | **DEAD** | 0/0/0 | — | учебная программа |
| `ziwago/app/data.py` | 17 | **DEAD** | 0/0/0 | — | — |
| `ziwago/app/tokenizer.py` | 108 | **SHIPPED** | 94/13/18 | — | BPE-токенизатор |
| `ziwago/chemistry/agents.py` | 216 | **SHIPPED** | 7688/0/360 | AgentLevels, ExtendedChemistry | химические агенты (аденозин, лактат…) |
| `ziwago/chemistry/modulators.py` | 143 | **SHIPPED** | 704/0/24 | — | скалярные модуляторы (DA/NE/ACh/5-HT) |
| `ziwago/chemistry/volume.py` | 94 | **SHIPPED** | 8448/0/396 | VolumeTransmission | объёмный обмен метаболитов |
| `ziwago/cognition/dialogue.py` | 254 | **SHIPPED** | 373/14/95 | ConversationalDialogueState | состояние диалога и эпизодическая история |
| `ziwago/cognition/language/content_form.py` | 226 | **PHANTOM** | 0/0/0 | ContentFormComposer | композитор content/form (не подключён) |
| `ziwago/cognition/language/parts/banks.py` | 223 | **SHIPPED** | 68/0/0 | — | — |
| `ziwago/cognition/language/parts/readout_math.py` | 384 | **SHIPPED** | 700/0/9 | — | — |
| `ziwago/cognition/language/parts/regulators.py` | 411 | **SHIPPED** | 280/0/0 | ReadoutRegulators | — |
| `ziwago/cognition/language/predictive_readout.py` | 189 | **SHIPPED** | 208/0/0 | PredictiveCodingCorticalReadout | предиктивный корковый ридеут |
| `ziwago/cognition/language/readout.py` | 690 | **SHIPPED** | 1572/0/158 | DendriticPlateauMatrixReadout | дендритно-плато ридеут + щит метапластичности |
| `ziwago/cognition/language/syntax.py` | 420 | **PHANTOM** | 0/0/0 | BA44AttractorSyntaxBridge | BA4/BA44 синтаксический мост |
| `ziwago/cognition/language/tdr.py` | 130 | **SHIPPED** | 0/0/7 | TokenCandidateIndex | TDR — выбор кандидат-индекса |
| `ziwago/cognition/memory/role_filler.py` | 155 | **SHIPPED** | 8/0/3 | ThetaGammaRoleFillerBinder | биндер ролей (θ-γ, подстановка заполнителя) |
| `ziwago/config.py` | 330 | **SHIPPED** | 128/0/0 | AttentionSchemaCfg, ChemistryCfg, HippocampusCfg | конфигурации |
| `ziwago/consciousness/attention_schema.py` | 135 | **SHIPPED** | 512/0/24 | AttentionSchema | схема внимания |
| `ziwago/consciousness/meta_head.py` | 295 | **SHIPPED** | 1540/0/72 | MetaHead, MetaOut | мета-голова (сознание) |
| `ziwago/consciousness/workspace.py` | 185 | **SHIPPED** | 2752/0/136 | GWOut, GlobalWorkspace | глобальное рабочее пространство |
| `ziwago/core/criticality.py` | 210 | **SHIPPED** | 490/0/24 | CriticalityMonitor | активная критичность (σ≈1) |
| `ziwago/core/dendrite.py` | 241 | **SHIPPED** | 2128/0/48 | DendriticTree | дендритное дерево |
| `ziwago/core/interoception.py` | 121 | **SHIPPED** | 256/0/12 | InteroOut, InteroceptiveField | интероцепция |
| `ziwago/core/multirate.py` | 749 | **SHIPPED** | 6480/0/288 | MinicolumnStateHierarchy, TickLadderScheduler | Benna–Fusi и иерархия мини-колонок |
| `ziwago/core/neuron.py` | 300 | **SHIPPED** | 3584/0/168 | USNeuronPool | пул спайковых нейронов (LIF) |
| `ziwago/core/sdr.py` | 643 | **SHIPPED** | 113/0/342 | CompositionalVSABPEEncoder, HashedCorticalSDREncoder | SDR/кодировщики |
| `ziwago/core/synapse.py` | 923 | **SHIPPED** | 23755/0/864 | GenesisSynapse | синапс и STDP |
| `ziwago/cortex/astrocyte.py` | 84 | **SHIPPED** | 1024/0/48 | AstrocyteMicrodomain | астроцитный микродомен |
| `ziwago/cortex/column.py` | 507 | **SHIPPED** | 4904/0/216 | CorticalColumn | корковая колонка |
| `ziwago/cortex/dcc.py` | 193 | **SHIPPED** | 511/0/0 | DevelopmentalReadinessController | DCC (готовность коры) |
| `ziwago/cortex/document_qa.py` | 530 | **DEAD** | 0/0/0 | — | QA по документу (grounded) |
| `ziwago/cortex/genesis_engine.py` | 1528 | **SHIPPED** | 710/5/52 | BA10CounterfactualInterventionSimulator, BidirectionalMeaningClosureVerifier, CorticalRelationalWorldGraph | движок генезиса (схемы, роли, эпос) |
| `ziwago/cortex/hierarchy.py` | 450 | **DEAD** | 0/0/0 | — | корковая иерархия |
| `ziwago/cortex/lfp.py` | 73 | **SHIPPED** | 512/0/24 | EphapticLFP | эфаптический LFP |
| `ziwago/cortex/moe_cortex.py` | 1510 | **SHIPPED** | 7338/0/100723 | AstroNMDALinearRecurrence, BlockSparse32x32Connector, DiscretePredictiveCodingBarrier | MoE-кора, часы, матричные операции |
| `ziwago/cortex/next16_core.py` | 1280 | **SHIPPED** | 1260842/0/57 | ConsciousnessMetricsDashboard, CorticalColumnarSubBankRouter, CrossDomainIsomorphismDreamer | BCM, Ларкум, щит, TDR-роутер, дашборд |
| `ziwago/cortex/son_reflection.py` | 356 | **SHIPPED** | 0/0/1 | SONReflectiveColumn | SON-рефлекторная колонка |
| `ziwago/cortex/spiking_language.py` | 182 | **SHIPPED** | 340/0/5 | SpikingLanguageLoop | спайковый языковой цикл |
| `ziwago/cortex/tiled_glia.py` | 154 | **SHIPPED** | 395/0/0 | TiledGlialControl | тайлованная глия (ACL) |
| `ziwago/emotion/limbic.py` | 164 | **SHIPPED** | 256/0/12 | LimbicOut, LimbicSystem | лимбический контур |
| `ziwago/evaluation/gates.py` | 1253 | **DEAD** | 0/0/0 | — | приёмочные ворота |
| `ziwago/homeostasis/controller.py` | 211 | **SHIPPED** | 24/0/0 | HomeostaticController | гомеостатический контроллер |
| `ziwago/learning/ewc.py` | 80 | **DEAD** | 0/0/0 | — | EWC-регуляризатор |
| `ziwago/learning/local_fisher.py` | 142 | **SHIPPED** | 234089/0/0 | SparseFisherAnchor | fisher-anchor |
| `ziwago/learning/mca.py` | 150 | **SHIPPED** | 233500/0/0 | MetaplasticSusceptibility | метапластическая чувствительность |
| `ziwago/learning/surrogate.py` | 90 | **DEAD** | 0/0/0 | — | суррогатные градиенты (запрещено в живом языке) |
| `ziwago/memory/dual_hippocampus.py` | 545 | **SHIPPED** | 11276/0/877 | ArbiterCfg, ArbiterDecision, DualHippoOut | дуальный гиппокамп (за флагом) |
| `ziwago/memory/hippocampus.py` | 1412 | **SHIPPED** | 83803/0/4450 | CompositionalThetaGammaIndexer, HippoOut, Hippocampus | спайковый гиппокамп EC→DG→CA3→CA1 |
| `ziwago/memory/laplace.py` | 67 | **DEAD** | 0/0/0 | — | спектр лапласовских шкал |
| `ziwago/memory/nsm.py` | 208 | **DEAD** | 0/0/0 | — | нарративная самомодель |
| `ziwago/memory/phase_vsa.py` | 357 | **DEAD** | 0/0/0 | — | θ-γ биндер историй |
| `ziwago/memory/tcc.py` | 458 | **DEAD** | 0/0/0 | — | тораидальный аттрактор отношений |
| `ziwago/memory/working.py` | 92 | **DEAD** | 0/0/0 | — | рабочая память |
| `ziwago/motor/pool.py` | 736 | **DEAD** | 0/0/0 | — | моторный пул |
| `ziwago/organism/admc.py` | 224 | **DEAD** | 0/0/0 | — | аллостатический контроллер любопытства |
| `ziwago/organism/flat.py` | 3487 | **SHIPPED** | 3163/68/181 | SixDialogCorporaStreamer, ZiwagoPureSynapticDialogueOrganismV14 | плоское ядро организма (монолит) |
| `ziwago/organism/integrated.py` | 674 | **SHIPPED** | 6040/0/279 | IntegratedOrganism, IntegrationConfig | интегрированный контур v20 |
| `ziwago/organism/physiology.py` | 406 | **SHIPPED** | 8408/0/372 | OrganismPhysiology, PhysiologyConfig, V20ChemState | физиология (единственный владелец химии) |
| `ziwago/organism/sleep.py` | 48 | **DEAD** | 0/0/0 | — | сон-обвязка |
| `ziwago/organism/verified.py` | 417 | **PHANTOM** | 0/0/0 | EmpiricalConfidenceCalibration, IndependentHypothesisSleep, SourceGroundingAuthority | проверяемый контур (evidence/authority) |
| `ziwago/organism_core.py` | 837 | **SHIPPED** | 4/1/1 | ZiwagoOrganism | обёртка ZiwagoOrganism (когнитивный цикл) |
| `ziwago/thalamus/relay.py` | 204 | **SHIPPED** | 1280/0/60 | ThalamicHO, ThalamusFO | таламус-реле |
| `ziwago/thalamus/trn.py` | 89 | **SHIPPED** | 256/0/12 | TRN | таламо-корковой TRN |

## Уровень 2 — механизмы (классы)

| Механизм | Файл | Статус | Экземпляров | Guard (условие включения) |
|---|---|---|---:|---|
| `BasalGanglia` | `ziwago/action/basal_ganglia.py` | **DEAD** | 0 | — |
| `StimulusResponsePair` | `ziwago/app/corpora.py` | **DEAD** | 0 | — |
| `NeuroChemistry` | `ziwago/chemistry/modulators.py` | **DEAD** | 0 | — |
| `RoleFiller` | `ziwago/cognition/language/content_form.py` | **DEAD** | 0 | — |
| `ReadoutBanks` | `ziwago/cognition/language/parts/banks.py` | **DEAD** | 0 | — |
| `ReadoutMath` | `ziwago/cognition/language/parts/readout_math.py` | **DEAD** | 0 | — |
| `GateApplication` | `ziwago/consciousness/meta_head.py` | **DEAD** | 0 | — |
| `Avalanche` | `ziwago/core/criticality.py` | **DEAD** | 0 | — |
| `BranchTree` | `ziwago/core/dendrite.py` | **DEAD** | 0 | — |
| `BennaFusiCascade` | `ziwago/core/multirate.py` | **DEAD** | 0 | — |
| `TickLadderStep` | `ziwago/core/multirate.py` | **DEAD** | 0 | — |
| `LCASparseEncoder` | `ziwago/core/sdr.py` | **DEAD** | 0 | — |
| `MorphoSemanticCorticalEncoder` | `ziwago/core/sdr.py` | **DEAD** | 0 | — |
| `DocumentEvidenceRanker` | `ziwago/cortex/document_qa.py` | **DEAD** | 0 | — |
| `OrderedEvidenceModel` | `ziwago/cortex/document_qa.py` | **DEAD** | 0 | — |
| `AxonalDelayPolychronousRouter` | `ziwago/cortex/hierarchy.py` | **DEAD** | 0 | — |
| `CorticalHierarchy` | `ziwago/cortex/hierarchy.py` | **DEAD** | 0 | — |
| `BA46SocraticInverseQuerySynthesizer` | `ziwago/cortex/next16_core.py` | **DEAD** | 0 | — |
| `SONReflectionVerdict` | `ziwago/cortex/son_reflection.py` | **DEAD** | 0 | — |
| `UnifiedCurriculumGates` | `ziwago/evaluation/gates.py` | **DEAD** | 0 | — |
| `UnifiedGateReport` | `ziwago/evaluation/gates.py` | **DEAD** | 0 | — |
| `EWCRegularizer` | `ziwago/learning/ewc.py` | **DEAD** | 0 | — |
| `_SpikeArctan` | `ziwago/learning/surrogate.py` | **DEAD** | 0 | — |
| `_SpikeSigmoid` | `ziwago/learning/surrogate.py` | **DEAD** | 0 | — |
| `LaplaceBank` | `ziwago/memory/laplace.py` | **DEAD** | 0 | — |
| `NarrativeSelfModule` | `ziwago/memory/nsm.py` | **DEAD** | 0 | — |
| `ThetaGammaStoryBinder` | `ziwago/memory/phase_vsa.py` | **DEAD** | 0 | — |
| `ToroidalRelationAttractor` | `ziwago/memory/tcc.py` | **DEAD** | 0 | — |
| `SilentWorkingMemory` | `ziwago/memory/working.py` | **DEAD** | 0 | — |
| `MotorPool` | `ziwago/motor/pool.py` | **DEAD** | 0 | — |
| `AllostaticCuriosityController` | `ziwago/organism/admc.py` | **DEAD** | 0 | — |
| `CheckpointCompatibilityError` | `ziwago/evaluation/gates.py` | **OFF** | 0 | not isinstance(col_to_tok_raw, list) or not isinstance(tok_to_col_raw, dict) |
| `PairedDialogueStreamer` | `ziwago/app/corpora.py` | **PHANTOM** | 1 | — |
| `ContentFormComposer` | `ziwago/cognition/language/content_form.py` | **PHANTOM** | 1 | — |
| `BA44AttractorSyntaxBridge` | `ziwago/cognition/language/syntax.py` | **PHANTOM** | 1 | — |
| `EmpiricalConfidenceCalibration` | `ziwago/organism/verified.py` | **PHANTOM** | 1 | — |
| `IndependentHypothesisSleep` | `ziwago/organism/verified.py` | **PHANTOM** | 1 | — |
| `SourceGroundingAuthority` | `ziwago/organism/verified.py` | **PHANTOM** | 1 | — |
| `ContextualActionGate` | `ziwago/action/contextual.py` | **SHIPPED** | 1 | — |
| `AgentLevels` | `ziwago/chemistry/agents.py` | **SHIPPED** | 1 | — |
| `ExtendedChemistry` | `ziwago/chemistry/agents.py` | **SHIPPED** | 1 | self.cfg.volume_chemistry_enabled |
| `VolumeTransmission` | `ziwago/chemistry/volume.py` | **SHIPPED** | 1 | self.cfg.volume_chemistry_enabled |
| `ReadoutRegulators` | `ziwago/cognition/language/parts/regulators.py` | **SHIPPED** | 1 | — |
| `PredictiveCodingCorticalReadout` | `ziwago/cognition/language/predictive_readout.py` | **SHIPPED** | 1 | — |
| `DendriticPlateauMatrixReadout` | `ziwago/cognition/language/readout.py` | **SHIPPED** | 1 | — |
| `TokenCandidateIndex` | `ziwago/cognition/language/tdr.py` | **SHIPPED** | 1 | idx is None or idx.encoder is not self.vsa_encoder |
| `ThetaGammaRoleFillerBinder` | `ziwago/cognition/memory/role_filler.py` | **SHIPPED** | 1 | — |
| `AttentionSchema` | `ziwago/consciousness/attention_schema.py` | **SHIPPED** | 1 | — |
| `MetaHead` | `ziwago/consciousness/meta_head.py` | **SHIPPED** | 1 | — |
| `GlobalWorkspace` | `ziwago/consciousness/workspace.py` | **SHIPPED** | 1 | — |
| `CriticalityMonitor` | `ziwago/core/criticality.py` | **SHIPPED** | 1 | self._criticality_monitor is None |
| `DendriticTree` | `ziwago/core/dendrite.py` | **SHIPPED** | 1 | — |
| `InteroceptiveField` | `ziwago/core/interoception.py` | **SHIPPED** | 1 | self.cfg.interoception_enabled |
| `MinicolumnStateHierarchy` | `ziwago/core/multirate.py` | **SHIPPED** | 1 | — |
| `TickLadderScheduler` | `ziwago/core/multirate.py` | **SHIPPED** | 1 | — |
| `USNeuronPool` | `ziwago/core/neuron.py` | **SHIPPED** | 1 | — |
| `CompositionalVSABPEEncoder` | `ziwago/core/sdr.py` | **SHIPPED** | 1 | — |
| `HashedCorticalSDREncoder` | `ziwago/core/sdr.py` | **SHIPPED** | 1 | — |
| `GenesisSynapse` | `ziwago/core/synapse.py` | **SHIPPED** | 1 | — |
| `AstrocyteMicrodomain` | `ziwago/cortex/astrocyte.py` | **SHIPPED** | 1 | — |
| `CorticalColumn` | `ziwago/cortex/column.py` | **SHIPPED** | 1 | — |
| `DevelopmentalReadinessController` | `ziwago/cortex/dcc.py` | **SHIPPED** | 1 | self.dcc_enabled |
| `BA10CounterfactualInterventionSimulator` | `ziwago/cortex/genesis_engine.py` | **SHIPPED** | 1 | — |
| `BidirectionalMeaningClosureVerifier` | `ziwago/cortex/genesis_engine.py` | **SHIPPED** | 1 | — |
| `CorticalRelationalWorldGraph` | `ziwago/cortex/genesis_engine.py` | **SHIPPED** | 1 | — |
| `EmergentOntologicalRoleDiscovery` | `ziwago/cortex/genesis_engine.py` | **SHIPPED** | 1 | — |
| `MDThalamusSynapticSpeechActRouter` | `ziwago/cortex/genesis_engine.py` | **SHIPPED** | 1 | — |
| `SyntacticValenceWaveGenerator` | `ziwago/cortex/genesis_engine.py` | **SHIPPED** | 1 | — |
| `ThetaGammaSpikingPhaseMultiplexer` | `ziwago/cortex/genesis_engine.py` | **SHIPPED** | 1 | — |
| `EphapticLFP` | `ziwago/cortex/lfp.py` | **SHIPPED** | 1 | — |
| `AstroNMDALinearRecurrence` | `ziwago/cortex/moe_cortex.py` | **SHIPPED** | 1 | — |
| `BlockSparse32x32Connector` | `ziwago/cortex/moe_cortex.py` | **SHIPPED** | 1 | — |
| `DiscretePredictiveCodingBarrier` | `ziwago/cortex/moe_cortex.py` | **SHIPPED** | 1 | — |
| `ReferenceInvariantLagPLVRouter` | `ziwago/cortex/moe_cortex.py` | **SHIPPED** | 1 | — |
| `SparselyGatedCorticalMoE` | `ziwago/cortex/moe_cortex.py` | **SHIPPED** | 1 | — |
| `ConsciousnessMetricsDashboard` | `ziwago/cortex/next16_core.py` | **SHIPPED** | 1 | — |
| `CorticalColumnarSubBankRouter` | `ziwago/cortex/next16_core.py` | **SHIPPED** | 1 | — |
| `CrossDomainIsomorphismDreamer` | `ziwago/cortex/next16_core.py` | **SHIPPED** | 1 | — |
| `DynamicNeuromodulatoryBCMGate` | `ziwago/cortex/next16_core.py` | **SHIPPED** | 1 | — |
| `FusiZenkeCascadeMetaplasticityShield` | `ziwago/cortex/next16_core.py` | **SHIPPED** | 1 | — |
| `KnowledgeDistillery` | `ziwago/cortex/next16_core.py` | **SHIPPED** | 1 | — |
| `TwoCompartmentLarkumMicrocircuit` | `ziwago/cortex/next16_core.py` | **SHIPPED** | 1 | — |
| `SONReflectiveColumn` | `ziwago/cortex/son_reflection.py` | **SHIPPED** | 1 | — |
| `SpikingLanguageLoop` | `ziwago/cortex/spiking_language.py` | **SHIPPED** | 1 | — |
| `TiledGlialControl` | `ziwago/cortex/tiled_glia.py` | **SHIPPED** | 1 | self.acl is None |
| `LimbicSystem` | `ziwago/emotion/limbic.py` | **SHIPPED** | 1 | self.cfg.limbic_enabled |
| `HomeostaticController` | `ziwago/homeostasis/controller.py` | **SHIPPED** | 1 | — |
| `SparseFisherAnchor` | `ziwago/learning/local_fisher.py` | **SHIPPED** | 1 | self.fisher_anchor_enabled |
| `MetaplasticSusceptibility` | `ziwago/learning/mca.py` | **SHIPPED** | 1 | self.mca_enabled |
| `DualHippocampus` | `ziwago/memory/dual_hippocampus.py` | **SHIPPED** | 1 | self.cfg.dual_hippocampus_enabled |
| `MergedLaplaceBank` | `ziwago/memory/dual_hippocampus.py` | **SHIPPED** | 1 | — |
| `NoveltyArbiter` | `ziwago/memory/dual_hippocampus.py` | **SHIPPED** | 1 | — |
| `ThetaClock` | `ziwago/memory/dual_hippocampus.py` | **SHIPPED** | 1 | — |
| `CompositionalThetaGammaIndexer` | `ziwago/memory/hippocampus.py` | **SHIPPED** | 1 | — |
| `Hippocampus` | `ziwago/memory/hippocampus.py` | **SHIPPED** | 1 | — |
| `SBCVSAShardedHippocampus` | `ziwago/memory/hippocampus.py` | **SHIPPED** | 1 | — |
| `SixDialogCorporaStreamer` | `ziwago/organism/flat.py` | **SHIPPED** | 1 | — |
| `ZiwagoPureSynapticDialogueOrganismV14` | `ziwago/organism/flat.py` | **SHIPPED** | 1 | — |
| `IntegratedOrganism` | `ziwago/organism/integrated.py` | **SHIPPED** | 1 | — |
| `OrganismPhysiology` | `ziwago/organism/physiology.py` | **SHIPPED** | 1 | — |
| `ZiwagoOrganism` | `ziwago/organism_core.py` | **SHIPPED** | 1 | — |
| `ThalamicHO` | `ziwago/thalamus/relay.py` | **SHIPPED** | 1 | — |
| `ThalamusFO` | `ziwago/thalamus/relay.py` | **SHIPPED** | 1 | — |
| `TRN` | `ziwago/thalamus/trn.py` | **SHIPPED** | 1 | — |

Итого по механизмам: DEAD — 31, OFF — 1, PHANTOM — 6, SHIPPED — 66
