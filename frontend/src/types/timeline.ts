export type TimelineNodeStatus =
  | 'completed'
  | 'active'
  | 'pending'
  | 'approval'
  | 'paused'

export type CorePageId =
  | 'workflow'
  | 'dialogue'
  | 'data-dictionary'
  | 'hypotheses'
  | 'uncertainties'
  | 'approval'
  | 'execution'
  | 'report'

export interface TimelineNodeData {
  id: string
  shortLabel: string
  title: string
  summary: string
  status: TimelineNodeStatus
  detailItems?: string[]
  relatedPage?: Exclude<CorePageId, 'home'>
}

export interface HypothesisPreviewNode {
  id: string
  displayLabel?: string
  label: string
  status: 'active' | 'newly_split' | 'observing' | 'weakened' | 'converged' | 'pruned' | 'pending' | 'draft'
  supportScore: number
  level?: number
}

export interface UncertaintyPreview {
  id: string
  title: string
  priority: 'P1' | 'P2' | 'P3'
  resolutionStatus: 'open' | 'tracking' | 'resolved'
  relatedHypotheses?: string[]
}

export interface ExperimentPreview {
  id: string
  targetHypothesis: string
  utility: number
  informationGain?: number
  performanceGain?: number
  status: 'recommended' | 'candidate' | 'approved' | 'completed'
  scientificQuestion?: string
}

export interface EvaluationPreview {
  experimentId?: string
  experimentType?: string
  baselinePearsonR?: number
  treatmentPearsonR?: number
  deltaPearsonR?: number
  deltaRmse?: number
  pgActual?: number
  verdict: 'supports' | 'weakens' | 'mixed'
  stable?: boolean
  runs?: ExperimentRunViewModel[]
  contrasts?: ExperimentContrastViewModel[]
  diagnostics?: unknown[]
}

export interface ExperimentRunViewModel {
  runId: string
  name?: string
  modelFamily?: string
  rmse?: number
  mae?: number
  pearsonR?: number
  skill?: number
  trainingMode?: string
  featureColumns?: string[]
  parameters?: Record<string, unknown>
}

export interface ExperimentContrastViewModel {
  contrastId: string
  name?: string
  kind?: string
  runIds?: string[]
  deltaRmse?: number
  deltaPearsonR?: number
  rmseCi?: number[] | null
  pearsonCi?: number[] | null
  note?: string
}

export interface DataCoverageViewModel {
  source?: string
  runId?: string
  expectedDays?: number
  observedDays?: number
  missingDays?: number
  coverageRatio?: number
  droppedGapWindows?: number
  interpolatedDays?: number
  note?: string
}

export interface RoundData {
  id: string
  roundNumber: number
  title: string
  subtitle: string
  stateLabel: string
  questionSummary: string
  conclusion?: string
  isCurrent: boolean
  isCollapsed: boolean
  nodes: TimelineNodeData[]
  hypotheses: HypothesisPreviewNode[]
  uncertainties: UncertaintyPreview[]
  experiments: ExperimentPreview[]
  evaluation?: EvaluationPreview
}

export interface MissionViewModel {
  scientificQuestion: string
  target: string
  questionType: string
  variables: {
    x?: string
    y?: string
    mCandidates: string[]
  }
  constraints: string[]
  metrics: string[]
  dataSources: Array<{
    id: string
    path?: string
    timeColumn?: string
    targetColumn?: string
  }>
  piGuidance: Array<{
    id: string
    timestamp?: string
    roundNumber?: number
    content: string
    source: string
  }>
  dialogueMessages: Array<{
    id: string
    role: 'system' | 'agent' | 'user'
    speaker: string
    content: string
    timestamp?: string
  }>
  expertStates: Array<{
    id: string
    label: string
    status: 'active' | 'completed' | 'running' | 'waiting'
    summary: string
  }>
}

export interface KnowledgeMemoryViewModel {
  rootQuestion: string
  treeSummary: {
    totalNodes: number
    activeCount: number
    prunedCount: number
    pendingCount: number
  }
  highlightedHypotheses: HypothesisPreviewNode[]
  allHypotheses: any[]
  uncertaintyQueue: Array<{
    id: string
    question: string
    priorityScore: number
    status: string
    estimatedResolutionRound?: number
  }>
  uncertaintyRecords: any[]
  experimentEntries: any[]
  ragSummaries: {
    project: string[]
    literature: string[]
    external: string[]
  }
}

export interface ExperimentEvaluationViewModel {
  recommendedExperimentId?: string
  executionPlanSummary?: string
  executedExperimentId?: string
  evaluatedRound?: number
  executionPlanExperimentId?: string
  candidateExperiments: any[]
  experimentEntries: any[]
  latestEvaluation?: EvaluationPreview
  modelParameters?: Record<string, unknown>
  tuningNarrative?: string
  tuningEntries?: Array<{
    refinementType?: string
    rationale?: string
    modelParameters?: Record<string, unknown>
    protocolNotes?: string[]
  }>
  metricComparison?: {
    experimentType?: string
    baselineRmse?: number
    treatmentRmse?: number
    baselinePearsonR?: number
    treatmentPearsonR?: number
    deltaPearsonR?: number
    deltaRmse?: number
    runs?: ExperimentRunViewModel[]
    contrasts?: ExperimentContrastViewModel[]
    diagnostics?: unknown[]
  }
  dataCoverage: DataCoverageViewModel[]
  visualizationPaths: string[]
  visualizationItems: Array<{
    path: string
    label: string
    previewUrl: string
    variant: 'baseline' | 'treatment' | 'artifact' | 'run'
    runId?: string
    runLabel?: string
  }>
}

export interface GovernanceDecisionsViewModel {
  processSummary: {
    currentPhase?: string
    currentStage?: string
    currentStep?: string
    autoContinue: boolean
    stopRequested: boolean
  }
  pendingApproval?: {
    candidateId?: string
    utilityScore?: number
    summary: string
  }
  approvals: any[]
  roundReviews: any[]
  auditTrail: any[]
  approvalComparisons: Array<{
    roundNumber: number
    requested: {
      candidateId: string
      utilityScore: number
      summary: string
      metrics: {
        informationGain: number
        performanceGain: number
        risk: number
        cost: number
        utility: number
      }
    }
    approved?: {
      candidateId: string
      utilityScore: number
      summary: string
      notes?: string
      metrics: {
        informationGain: number
        performanceGain: number
        risk: number
        cost: number
        utility: number
      }
    }
    review?: {
      experimentId?: string
      summary: string
    }
    notes?: string
    hypothesisAssessments: any[]
    disagreementUpdates: any[]
  }>
  stopHistory: any[]
}

export interface ProcessMonitorViewModel {
  currentRound: number
  currentPhase?: string
  currentStage?: string
  currentStep?: string
  progressPercentage: number
  pendingApprovalCandidateId?: string
  stages: Array<{
    id: string
    label: string
    status: 'completed' | 'running' | 'pending'
    timestamp?: string
    steps?: Array<{
      id: string
      label: string
      status: 'completed' | 'running' | 'pending'
    }>
  }>
  currentStepDetail: {
    title: string
    status: string
    candidateCount: number
    recommendedExperimentId?: string
  }
  recentLogs: Array<{
    id: string
    timestamp?: string
    summary: string
  }>
}

export interface RoundReportViewModel {
  roundNumber: number
  recommendedAction: 'next_round' | 'adjust' | 'stop'
  sourceExperimentId?: string
  sourceUpdatedAt?: string
  threeLayerConclusion?: ThreeLayerConclusionViewModel
  summary: {
    experimentType?: string
    baselinePearsonR?: number
    treatmentPearsonR?: number
    deltaPearsonR?: number
    baselineRmse?: number
    treatmentRmse?: number
    deltaRmse?: number
    runs?: ExperimentRunViewModel[]
    contrasts?: ExperimentContrastViewModel[]
  }
  dataCoverage: DataCoverageViewModel[]
  scientificConclusions: string[]
  highlightedHypotheses: HypothesisPreviewNode[]
  unresolvedQuestions: string[]
  decisionOptions: string[]
  iterationEvidence?: {
    sourceRound: number
    inputSources: string[]
    validations: Array<{
      itemId: string
      label: string
      passed: boolean
      detail?: string
    }>
    previousFocus?: string
    nextFocus?: string
    previousExperimentId?: string
    nextExperimentId?: string
  }
  candidateEvolution?: {
    previousId?: string
    nextId?: string
    previousFocus?: string
    nextFocus?: string
    previousTreatment: string[]
    nextTreatment: string[]
    summary: string
  }
  modelParameters?: Record<string, unknown>
  tuningNarrative?: string
  tuningEntries?: Array<{
    refinementType?: string
    rationale?: string
    modelParameters?: Record<string, unknown>
    protocolNotes?: string[]
  }>
}

export interface ThreeLayerConclusionExperimentLayer {
  experimentId: string
  designSummary: string
  probeAxis?: string
  forecastHorizonDays?: number
  baselineRmse?: number
  treatmentRmse?: number
  baselinePearsonR?: number
  treatmentPearsonR?: number
  skillDelta?: number
  decisive: boolean
  experimentType?: string
  runs?: ExperimentRunViewModel[]
  contrasts?: ExperimentContrastViewModel[]
  diagnostics?: unknown[]
}

export interface ThreeLayerConclusionHypothesisRow {
  hypothesisId: string
  displayHypothesisId: string
  statement: string
  predictedDirection?: string
  predictedRange?: [number, number]
  actualDelta?: number
  directionMatched?: boolean | string
  magnitudeMatched?: boolean | string
  conclusion: string
  supportAfter?: number
}

export interface ThreeLayerConclusionScientificLayer {
  mainQuestion: string
  answer: string
  pathQuestion?: string
  pathAnswer?: string
  evidenceText?: string
}

export interface ConclusionDataLayer {
  rmseAttribution?: string
  pearsonAttribution?: string
  skillDeltaMeaning?: string
  anomalies: string[]
  nextFocus?: string
  chartAnalyses?: ConclusionChartAnalysis[]
  comparisonAnalysis?: string
}

export interface ConclusionChartAnalysis {
  chartName: string
  chartRole: string
  chartImagePath?: string
  description: string
  keyObservations: string[]
  anomalyOrInsight?: string
}

export interface ConclusionNextRoundSuggestion {
  evidenceSummary?: string
  resolvedUncertaintiesThisRound: string[]
  unresolvedUncertaintiesTodo: string[]
  remainingUncertaintyAnalysis?: string
  recommendation?: 'continue' | 'adjust' | 'stop'
  piDecisionAdvice?: string
  experimentDesignAdvice?: string
  hypothesisSpaceAdvice?: string
  notes: string[]
}

export interface ConclusionTrackingLayer {
  auditItems: string[]
  sources: string[]
  snapshotRefs: string[]
}

export interface ThreeLayerConclusionViewModel {
  experimentLayer: ThreeLayerConclusionExperimentLayer
  hypothesisLayer: ThreeLayerConclusionHypothesisRow[]
  scientificLayer: ThreeLayerConclusionScientificLayer
  dataLayer?: ConclusionDataLayer
  trackingLayer?: ConclusionTrackingLayer
  hypothesisLayerSummary?: string
  overallSummary?: string
  nextRoundSuggestion?: ConclusionNextRoundSuggestion
}

export interface ApprovalOverlayViewModel {
  roundNumber: number
  isPending: boolean
  recommendation: {
    candidateId?: string
    reason: string
    evidence: string[]
  }
  candidates: Array<{
    experimentId: string
    scientificQuestion: string
    informationGain: number
    performanceGain: number
    risk: number
    cost: number
    utility: number
    recommended: boolean
    relatedUncertainties: string[]
    testedHypotheses: string[]
    controlVariables: string[]
    treatmentVariables: string[]
    experimentMode: 'baseline' | 'comparative'
    hasFeatureDifference: boolean
  }>
  competitionChecklist: Array<{
    label: string
    status: 'ready' | 'partial'
    summary: string
  }>
  iterationSignals: string[]
}

export interface DecisionLogViewerViewModel {
  records: Array<{
    id: string
    timestamp?: string
    roundNumber?: number
    type: string
    actor: string
    summary: string
    details: string[]
  }>
  availableTypes: string[]
  availableActors: string[]
}

export interface DataManagerViewModel {
  sources: Array<{
    id: string
    path?: string
    timeColumn?: string
    targetColumn?: string
    role: string
  }>
  variableDictionary: Array<{
    field: string
    meaning: string
    source: string
  }>
  resourceBudget: Array<{
    label: string
    value: string
  }>
  qualityChecks: Array<{
    id: string
    label: string
    status: 'pass' | 'warning'
    note: string
  }>
}

export interface KnowledgeBaseManagerViewModel {
  collections: Array<{
    id: string
    label: string
    count: number
    status: 'ready' | 'partial'
    summary: string
  }>
  ragEntries: Array<{
    id: string
    stage: string
    summary: string
    hypotheses: string[]
    uncertainties: string[]
  }>
  uploads: Array<{
    id: string
    fileName: string
    fileType: string
    uploadedAt?: string
    extractionStatus: string
    syncStatus: string
    storagePath: string
    note: string
  }>
  modelUsage: Array<{
    scope: string
    model: string
    purpose: string
  }>
  tags: string[]
}

export interface FrontendStateSnapshot {
  task: any
  process: any
  decisionLog: any
  hypothesisTree: any
  uncertainties: any
  experimentMemory: any
  candidateExperiments: any
  plannerInput: any
  plannerOutput: any | null
  sessionStatus?: any
  uploadManifest?: any
  modelUsage?: any
}

export interface FrontendViewModels {
  rounds: RoundData[]
  currentRoundNumber: number
  currentRoundId: string
  currentNodeId: string | null
  progressPercentage: number
  processStage: string
  mission: MissionViewModel
  knowledgeMemory: KnowledgeMemoryViewModel
  experimentEvaluation: ExperimentEvaluationViewModel
  governance: GovernanceDecisionsViewModel
  processMonitor: ProcessMonitorViewModel
  roundReport: RoundReportViewModel
  approvalOverlay: ApprovalOverlayViewModel
  decisionLogViewer: DecisionLogViewerViewModel
  dataManager: DataManagerViewModel
  knowledgeBaseManager: KnowledgeBaseManagerViewModel
}

export interface TimelineDataBundle {
  snapshot: FrontendStateSnapshot
  viewModels: FrontendViewModels
  sourceMode: 'api' | 'empty'
  refreshedAt: string
}
