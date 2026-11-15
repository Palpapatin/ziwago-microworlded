# Карта модулей Ziwago18 — «заявлено vs работает»

> **УСТАРЕВШАЯ ГЕНЕРИРОВАННАЯ ТАБЛИЦА (13.11.2026).** Количество вызовов не
> доказывает причинный эффект, а владельцы и точки проводки изменились. Не
> используйте статусы ниже для удаления или подключения модулей. Свежая
> перепроверка: [MODULE_LIVENESS_RECHECK_2026-11-14.md](MODULE_LIVENESS_RECHECK_2026-11-14.md).

Сгенерировано из измерений: `tools/probes/module_map_probe.py` → `tools/probes/build_module_map.py`. Классификация НЕ выдумана: статус выведен из числа реальных вызовов функций в фазах обучение / чат / прямая генерация.

## Уровень 1 — файлы

| Статус | Файлов | Строк |
|---|---:|---:|
| SHIPPED | 26 | 16563 |
| PHANTOM | 3 | 559 |
| OFF | 0 | 0 |
| DEAD | 41 | 9183 |
| **Итого** | **70** | **26305** |

| Файл | Строк | Статус | Вызовы (train/chat/gen) | Экземпляров | Роль |
|---|---:|---|---:|---:|---|
| `ziwago/action/basal_ganglia.py` | 123 | **DEAD** | 0/0/0 | — | маршрутизация действия (D1, выбора ответа) |
| `ziwago/action/contextual.py` | 396 | **DEAD** | 0/0/0 | — | контекстная политика действия |
| `ziwago/app/corpora.py` | 139 | **PHANTOM** | 0/0/0 | PairedDialogueStreamer | поток диалоговых корпусов |
| `ziwago/app/curriculum.py` | 66 | **DEAD** | 0/0/0 | — | учебная программа |
| `ziwago/app/data.py` | 17 | **DEAD** | 0/0/0 | — | — |
| `ziwago/app/tokenizer.py` | 108 | **SHIPPED** | 94/65/18 | — | BPE-токенизатор |
| `ziwago/chemistry/agents.py` | 216 | **DEAD** | 0/0/0 | — | химические агенты (аденозин, лактат…) |
| `ziwago/chemistry/modulators.py` | 143 | **SHIPPED** | 120/0/0 | — | скалярные модуляторы (DA/NE/ACh/5-HT) |
| `ziwago/chemistry/volume.py` | 94 | **DEAD** | 0/0/0 | — | объёмный обмен метаболитов |
| `ziwago/cognition/dialogue.py` | 254 | **SHIPPED** | 373/161/95 | ConversationalDialogueState | состояние диалога и эпизодическая история |
| `ziwago/cognition/language/content_form.py` | 194 | **DEAD** | 0/0/0 | — | композитор content/form (не подключён) |
| `ziwago/cognition/language/predictive_readout.py` | 189 | **SHIPPED** | 208/378/186 | PredictiveCodingCorticalReadout | предиктивный корковый ридеут |
| `ziwago/cognition/language/readout.py` | 1230 | **SHIPPED** | 1609/1525/756 | DendriticPlateauMatrixReadout | дендритно-плато ридеут + щит метапластичности |
| `ziwago/cognition/language/syntax.py` | 260 | **SHIPPED** | 0/45/33 | BA44AttractorSyntaxBridge | BA4/BA44 синтаксический мост |
| `ziwago/cognition/language/tdr.py` | 130 | **DEAD** | 0/0/0 | — | TDR — выбор кандидат-индекса |
| `ziwago/cognition/memory/role_filler.py` | 113 | **SHIPPED** | 8/3/3 | ThetaGammaRoleFillerBinder | биндер ролей (θ-γ, подстановка заполнителя) |
| `ziwago/config.py` | 330 | **PHANTOM** | 0/0/0 | AttentionSchemaCfg, ChemistryCfg, MetaHeadCfg | конфигурации |
| `ziwago/consciousness/attention_schema.py` | 90 | **PHANTOM** | 0/0/0 | AttentionSchema | схема внимания |
| `ziwago/consciousness/meta_head.py` | 145 | **SHIPPED** | 0/3048/1542 | MetaHead | мета-голова (сознание) |
| `ziwago/consciousness/metrics.py` | 131 | **DEAD** | 0/0/0 | — | метрики сознания (дубликат next16_core) |
| `ziwago/consciousness/workspace.py` | 125 | **SHIPPED** | 0/508/257 | GlobalWorkspace | глобальное рабочее пространство |
| `ziwago/core/criticality.py` | 150 | **DEAD** | 0/0/0 | — | активная критичность (σ≈1) |
| `ziwago/core/dendrite.py` | 205 | **SHIPPED** | 352/384/192 | DendriticTree | дендритное дерево |
| `ziwago/core/e2e_predictive.py` | 145 | **DEAD** | 0/0/0 | — | предиктивное кодирование e2e |
| `ziwago/core/interoception.py` | 121 | **DEAD** | 0/0/0 | — | интероцепция |
| `ziwago/core/multirate.py` | 482 | **SHIPPED** | 256/768/384 | BennaFusiCascade, MinicolumnStateHierarchy, TickLadderScheduler | Benna–Fusi и иерархия мини-колонок |
| `ziwago/core/neuron.py` | 300 | **SHIPPED** | 896/2688/1344 | USNeuronPool | пул спайковых нейронов (LIF) |
| `ziwago/core/sdr.py` | 607 | **SHIPPED** | 113/0/0 | CompositionalVSABPEEncoder, HashedCorticalSDREncoder | SDR/кодировщики |
| `ziwago/core/synapse.py` | 906 | **SHIPPED** | 962/2300/1148 | GenesisSynapse | синапс и STDP |
| `ziwago/cortex/astrocyte.py` | 53 | **DEAD** | 0/0/0 | — | астроцитный микродомен |
| `ziwago/cortex/column.py` | 456 | **SHIPPED** | 240/384/192 | CorticalColumn | корковая колонка |
| `ziwago/cortex/dcc.py` | 193 | **DEAD** | 0/0/0 | — | DCC (готовность коры) |
| `ziwago/cortex/document_qa.py` | 530 | **DEAD** | 0/0/0 | — | QA по документу (grounded) |
| `ziwago/cortex/genesis_engine.py` | 1528 | **SHIPPED** | 710/995/460 | BA10CounterfactualInterventionSimulator, BidirectionalMeaningClosureVerifier, CorticalRelationalWorldGraph | движок генезиса (схемы, роли, эпос) |
| `ziwago/cortex/hierarchy.py` | 441 | **DEAD** | 0/0/0 | — | корковая иерархия |
| `ziwago/cortex/lfp.py` | 73 | **SHIPPED** | 128/384/192 | EphapticLFP | эфаптический LFP |
| `ziwago/cortex/moe_cortex.py` | 1470 | **SHIPPED** | 595/18897691/9488052 | AstroNMDALinearRecurrence, BlockSparse32x32Connector, DiscretePredictiveCodingBarrier | MoE-кора, часы, матричные операции |
| `ziwago/cortex/next16_core.py` | 1280 | **SHIPPED** | 1260914/10847/5358 | ConsciousnessMetricsDashboard, CorticalColumnarSubBankRouter, CrossDomainIsomorphismDreamer | BCM, Ларкум, щит, TDR-роутер, дашборд |
| `ziwago/cortex/son_reflection.py` | 304 | **SHIPPED** | 0/510/259 | SONReflectiveColumn | SON-рефлекторная колонка |
| `ziwago/cortex/spiking_language.py` | 182 | **SHIPPED** | 340/2093/1037 | SpikingLanguageLoop | спайковый языковой цикл |
| `ziwago/cortex/tiled_glia.py` | 154 | **DEAD** | 0/0/0 | — | тайлованная глия (ACL) |
| `ziwago/embodied/arena.py` | 263 | **DEAD** | 0/0/0 | — | арена телесного поиска |
| `ziwago/emotion/limbic.py` | 164 | **DEAD** | 0/0/0 | — | лимбический контур |
| `ziwago/evaluation/gates.py` | 1253 | **SHIPPED** | 0/41/41 | — | приёмочные ворота |
| `ziwago/homeostasis/controller.py` | 111 | **DEAD** | 0/0/0 | — | гомеостатический контроллер |
| `ziwago/learning/ewc.py` | 75 | **DEAD** | 0/0/0 | — | EWC-регуляризатор |
| `ziwago/learning/local_fisher.py` | 128 | **DEAD** | 0/0/0 | — | fisher-anchor |
| `ziwago/learning/mca.py` | 131 | **DEAD** | 0/0/0 | — | метапластическая чувствительность |
| `ziwago/learning/phase.py` | 197 | **DEAD** | 0/0/0 | — | фазовое обучение |
| `ziwago/learning/predictive.py` | 55 | **DEAD** | 0/0/0 | — | предиктивная голова |
| `ziwago/learning/surrogate.py` | 78 | **DEAD** | 0/0/0 | — | суррогатные градиенты (запрещено в живом языке) |
| `ziwago/learning/sync.py` | 100 | **DEAD** | 0/0/0 | — | синхронизация пол半球 |
| `ziwago/memory/dual_hippocampus.py` | 496 | **DEAD** | 0/0/0 | — | дуальный гиппокамп (за флагом) |
| `ziwago/memory/hippocampus.py` | 1124 | **DEAD** | 0/0/0 | — | спайковый гиппокамп EC→DG→CA3→CA1 |
| `ziwago/memory/laplace.py` | 51 | **DEAD** | 0/0/0 | — | спектр лапласовских шкал |
| `ziwago/memory/nsm.py` | 208 | **DEAD** | 0/0/0 | — | нарративная самомодель |
| `ziwago/memory/phase_vsa.py` | 269 | **DEAD** | 0/0/0 | — | θ-γ биндер историй |
| `ziwago/memory/tcc.py` | 402 | **DEAD** | 0/0/0 | — | тораидальный аттрактор отношений |
| `ziwago/memory/working.py` | 57 | **DEAD** | 0/0/0 | — | рабочая память |
| `ziwago/motor/pool.py` | 733 | **DEAD** | 0/0/0 | — | моторный пул |
| `ziwago/motor/syllabary_decoder.py` | 412 | **DEAD** | 0/0/0 | — | декодер слогов |
| `ziwago/organism/admc.py` | 195 | **DEAD** | 0/0/0 | — | аллостатический контроллер любопытства |
| `ziwago/organism/flat.py` | 3420 | **SHIPPED** | 3111/18990/9488 | SixDialogCorporaStreamer, ZiwagoPureSynapticDialogueOrganismV14 | плоское ядро организма (монолит) |
| `ziwago/organism/integrated.py` | 556 | **DEAD** | 0/0/0 | — | интегрированный контур v20 |
| `ziwago/organism/physiology.py` | 406 | **SHIPPED** | 180/0/0 | OrganismPhysiology, PhysiologyConfig, V20ChemState | физиология (единственный владелец химии) |
| `ziwago/organism/sleep.py` | 48 | **DEAD** | 0/0/0 | — | сон-обвязка |
| `ziwago/organism/verified.py` | 417 | **SHIPPED** | 0/79/54 | EmpiricalConfidenceCalibration, IndependentHypothesisSleep, SourceGroundingAuthority | проверяемый контур (evidence/authority) |
| `ziwago/organism_core.py` | 707 | **SHIPPED** | 4/2/1 | ZiwagoOrganism | обёртка ZiwagoOrganism (когнитивный цикл) |
| `ziwago/thalamus/relay.py` | 112 | **DEAD** | 0/0/0 | — | таламус-реле |
| `ziwago/thalamus/trn.py` | 64 | **DEAD** | 0/0/0 | — | таламо-корковой TRN |

## Уровень 2 — механизмы (классы)

| Механизм | Файл | Статус | Экземпляров | Guard (условие включения) |
|---|---|---|---:|---|
| `BasalGanglia` | `ziwago/action/basal_ganglia.py` | **DEAD** | 0 | — |
| `ContextualActionGate` | `ziwago/action/contextual.py` | **DEAD** | 0 | — |
| `StimulusResponsePair` | `ziwago/app/corpora.py` | **DEAD** | 0 | — |
| `AgentLevels` | `ziwago/chemistry/agents.py` | **DEAD** | 0 | — |
| `NeuroChemistry` | `ziwago/chemistry/modulators.py` | **DEAD** | 0 | — |
| `ContentFormComposer` | `ziwago/cognition/language/content_form.py` | **DEAD** | 0 | — |
| `RoleFiller` | `ziwago/cognition/language/content_form.py` | **DEAD** | 0 | — |
| `ConsciousnessDashboard` | `ziwago/consciousness/metrics.py` | **DEAD** | 0 | — |
| `Avalanche` | `ziwago/core/criticality.py` | **DEAD** | 0 | — |
| `CriticalityMonitor` | `ziwago/core/criticality.py` | **DEAD** | 0 | — |
| `BranchTree` | `ziwago/core/dendrite.py` | **DEAD** | 0 | — |
| `EndToEndPredictiveCodingLoop` | `ziwago/core/e2e_predictive.py` | **DEAD** | 0 | — |
| `TickLadderStep` | `ziwago/core/multirate.py` | **DEAD** | 0 | — |
| `LCASparseEncoder` | `ziwago/core/sdr.py` | **DEAD** | 0 | — |
| `MorphoSemanticCorticalEncoder` | `ziwago/core/sdr.py` | **DEAD** | 0 | — |
| `AstrocyteMicrodomain` | `ziwago/cortex/astrocyte.py` | **DEAD** | 0 | — |
| `DocumentEvidenceRanker` | `ziwago/cortex/document_qa.py` | **DEAD** | 0 | — |
| `OrderedEvidenceModel` | `ziwago/cortex/document_qa.py` | **DEAD** | 0 | — |
| `AxonalDelayPolychronousRouter` | `ziwago/cortex/hierarchy.py` | **DEAD** | 0 | — |
| `CorticalHierarchy` | `ziwago/cortex/hierarchy.py` | **DEAD** | 0 | — |
| `BA46SocraticInverseQuerySynthesizer` | `ziwago/cortex/next16_core.py` | **DEAD** | 0 | — |
| `SONReflectionVerdict` | `ziwago/cortex/son_reflection.py` | **DEAD** | 0 | — |
| `EmbodiedForagingArena` | `ziwago/embodied/arena.py` | **DEAD** | 0 | — |
| `UnifiedCurriculumGates` | `ziwago/evaluation/gates.py` | **DEAD** | 0 | — |
| `UnifiedGateReport` | `ziwago/evaluation/gates.py` | **DEAD** | 0 | — |
| `HomeostaticController` | `ziwago/homeostasis/controller.py` | **DEAD** | 0 | — |
| `EWCRegularizer` | `ziwago/learning/ewc.py` | **DEAD** | 0 | — |
| `PhaseVSA` | `ziwago/learning/phase.py` | **DEAD** | 0 | — |
| `ThetaGammaBinder` | `ziwago/learning/phase.py` | **DEAD** | 0 | — |
| `PredictiveHead` | `ziwago/learning/predictive.py` | **DEAD** | 0 | — |
| `_SpikeArctan` | `ziwago/learning/surrogate.py` | **DEAD** | 0 | — |
| `_SpikeSigmoid` | `ziwago/learning/surrogate.py` | **DEAD** | 0 | — |
| `HemisphereSync` | `ziwago/learning/sync.py` | **DEAD** | 0 | — |
| `MergedLaplaceBank` | `ziwago/memory/dual_hippocampus.py` | **DEAD** | 0 | — |
| `NoveltyArbiter` | `ziwago/memory/dual_hippocampus.py` | **DEAD** | 0 | — |
| `ThetaClock` | `ziwago/memory/dual_hippocampus.py` | **DEAD** | 0 | — |
| `CompositionalThetaGammaIndexer` | `ziwago/memory/hippocampus.py` | **DEAD** | 0 | — |
| `Hippocampus` | `ziwago/memory/hippocampus.py` | **DEAD** | 0 | — |
| `SBCVSAShardedHippocampus` | `ziwago/memory/hippocampus.py` | **DEAD** | 0 | — |
| `LaplaceBank` | `ziwago/memory/laplace.py` | **DEAD** | 0 | — |
| `NarrativeSelfModule` | `ziwago/memory/nsm.py` | **DEAD** | 0 | — |
| `ThetaGammaStoryBinder` | `ziwago/memory/phase_vsa.py` | **DEAD** | 0 | — |
| `ToroidalRelationAttractor` | `ziwago/memory/tcc.py` | **DEAD** | 0 | — |
| `SilentWorkingMemory` | `ziwago/memory/working.py` | **DEAD** | 0 | — |
| `MotorPool` | `ziwago/motor/pool.py` | **DEAD** | 0 | — |
| `BGPQSyllabaryDecoder` | `ziwago/motor/syllabary_decoder.py` | **DEAD** | 0 | — |
| `AllostaticCuriosityController` | `ziwago/organism/admc.py` | **DEAD** | 0 | — |
| `IntegratedOrganism` | `ziwago/organism/integrated.py` | **DEAD** | 0 | — |
| `ThalamicHO` | `ziwago/thalamus/relay.py` | **DEAD** | 0 | — |
| `ThalamusFO` | `ziwago/thalamus/relay.py` | **DEAD** | 0 | — |
| `TRN` | `ziwago/thalamus/trn.py` | **DEAD** | 0 | — |
| `ExtendedChemistry` | `ziwago/chemistry/agents.py` | **OFF** | 0 | self.cfg.volume_chemistry_enabled |
| `VolumeTransmission` | `ziwago/chemistry/volume.py` | **OFF** | 0 | self.cfg.volume_chemistry_enabled |
| `TokenCandidateIndex` | `ziwago/cognition/language/tdr.py` | **OFF** | 0 | idx is None or idx.encoder is not self.vsa_encoder |
| `InteroceptiveField` | `ziwago/core/interoception.py` | **OFF** | 0 | self.cfg.interoception_enabled |
| `DevelopmentalReadinessController` | `ziwago/cortex/dcc.py` | **OFF** | 0 | self._dcc is None |
| `TiledGlialControl` | `ziwago/cortex/tiled_glia.py` | **OFF** | 0 | self._acl is None |
| `LimbicSystem` | `ziwago/emotion/limbic.py` | **OFF** | 0 | self.cfg.limbic_enabled |
| `CheckpointCompatibilityError` | `ziwago/evaluation/gates.py` | **OFF** | 0 | not isinstance(col_to_tok_raw, list) or not isinstance(tok_to_col_raw, dict) |
| `SparseFisherAnchor` | `ziwago/learning/local_fisher.py` | **OFF** | 0 | bool(getattr(self, 'fisher_anchor_enabled', False)) |
| `MetaplasticSusceptibility` | `ziwago/learning/mca.py` | **OFF** | 0 | bool(getattr(self, 'mca_enabled', False)) |
| `DualHippocampus` | `ziwago/memory/dual_hippocampus.py` | **OFF** | 0 | self.cfg.dual_hippocampus_enabled |
| `PairedDialogueStreamer` | `ziwago/app/corpora.py` | **PHANTOM** | 1 | — |
| `AttentionSchema` | `ziwago/consciousness/attention_schema.py` | **PHANTOM** | 1 | — |
| `PredictiveCodingCorticalReadout` | `ziwago/cognition/language/predictive_readout.py` | **SHIPPED** | 1 | — |
| `DendriticPlateauMatrixReadout` | `ziwago/cognition/language/readout.py` | **SHIPPED** | 1 | — |
| `BA44AttractorSyntaxBridge` | `ziwago/cognition/language/syntax.py` | **SHIPPED** | 1 | — |
| `ThetaGammaRoleFillerBinder` | `ziwago/cognition/memory/role_filler.py` | **SHIPPED** | 1 | — |
| `MetaHead` | `ziwago/consciousness/meta_head.py` | **SHIPPED** | 1 | — |
| `GlobalWorkspace` | `ziwago/consciousness/workspace.py` | **SHIPPED** | 1 | — |
| `DendriticTree` | `ziwago/core/dendrite.py` | **SHIPPED** | 1 | — |
| `BennaFusiCascade` | `ziwago/core/multirate.py` | **SHIPPED** | 1 | — |
| `MinicolumnStateHierarchy` | `ziwago/core/multirate.py` | **SHIPPED** | 1 | — |
| `TickLadderScheduler` | `ziwago/core/multirate.py` | **SHIPPED** | 1 | — |
| `USNeuronPool` | `ziwago/core/neuron.py` | **SHIPPED** | 1 | — |
| `CompositionalVSABPEEncoder` | `ziwago/core/sdr.py` | **SHIPPED** | 1 | — |
| `HashedCorticalSDREncoder` | `ziwago/core/sdr.py` | **SHIPPED** | 1 | — |
| `GenesisSynapse` | `ziwago/core/synapse.py` | **SHIPPED** | 1 | — |
| `CorticalColumn` | `ziwago/cortex/column.py` | **SHIPPED** | 1 | — |
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
| `SixDialogCorporaStreamer` | `ziwago/organism/flat.py` | **SHIPPED** | 1 | — |
| `ZiwagoPureSynapticDialogueOrganismV14` | `ziwago/organism/flat.py` | **SHIPPED** | 1 | — |
| `OrganismPhysiology` | `ziwago/organism/physiology.py` | **SHIPPED** | 1 | — |
| `EmpiricalConfidenceCalibration` | `ziwago/organism/verified.py` | **SHIPPED** | 1 | — |
| `IndependentHypothesisSleep` | `ziwago/organism/verified.py` | **SHIPPED** | 1 | — |
| `SourceGroundingAuthority` | `ziwago/organism/verified.py` | **SHIPPED** | 1 | — |
| `ZiwagoOrganism` | `ziwago/organism_core.py` | **SHIPPED** | 1 | — |

Итого по механизмам: DEAD — 51, OFF — 11, PHANTOM — 2, SHIPPED — 44
