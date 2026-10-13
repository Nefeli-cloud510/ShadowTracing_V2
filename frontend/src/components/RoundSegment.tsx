import type {
  ExperimentPreview,
  HypothesisPreviewNode,
  RoundData,
  TimelineNodeData,
  TimelineNodeStatus,
  UncertaintyPreview,
} from '../types/timeline'
import { TimelineNode } from './TimelineNode'

interface RoundSegmentProps {
  round: RoundData
  selectedNodeId: string
  onSelectNode: (node: TimelineNodeData) => void
  scale: number
}

function renderUncertaintyItem(item: UncertaintyPreview) {
  return `${item.id} · ${item.title} · ${item.priority}`
}

function getLeadExperiment(experiments: ExperimentPreview[]) {
  return experiments.find((item) => item.status === 'recommended' || item.status === 'approved')
}

function getLeadHypothesis(hypotheses: HypothesisPreviewNode[]) {
  return hypotheses[0]
}

function hasReached(status?: TimelineNodeStatus) {
  return status === 'completed' || status === 'active' || status === 'approval'
}

export function RoundSegment({
  round,
  selectedNodeId,
  onSelectNode,
  scale,
}: RoundSegmentProps) {
  const leadExperiment = getLeadExperiment(round.experiments)
  const leadHypothesis = getLeadHypothesis(round.hypotheses)
  const leadUncertainty = round.uncertainties[0]
  const questionNode = round.nodes.find((node) => node.id === 'Q')
  const hypothesisNode = round.nodes.find((node) => node.id === 'H')
  const uncertaintyNode = round.nodes.find((node) => node.id === 'U')
  const experimentNode = round.nodes.find((node) => node.id === 'E')
  const evaluationNode = round.nodes.find((node) => node.id === 'A')

  const showQuestionCard = hasReached(questionNode?.status)
  const showFeatureCard = hasReached(experimentNode?.status)
    ? Boolean(leadExperiment)
    : hasReached(hypothesisNode?.status)
      ? Boolean(leadHypothesis)
      : false
  const showResultCard = hasReached(evaluationNode?.status)
    ? Boolean(round.evaluation)
    : hasReached(uncertaintyNode?.status)
      ? Boolean(leadUncertainty)
      : false

  const segmentWidth = 1220 * scale
  const segmentHeight = 520 * scale

  return (
    <section
      className={['round-segment', round.isCollapsed ? 'round-segment--collapsed' : ''].filter(Boolean).join(' ')}
      aria-label={round.title}
      style={{ width: `${segmentWidth}px`, minHeight: `${segmentHeight}px` }}
    >
      <div className="round-segment__header">
        <span className="round-segment__title">{round.title}</span>
        <span className="round-segment__state">{round.isCurrent ? round.stateLabel : '已完成'}</span>
      </div>
      <div
        className="round-segment__scale"
        style={{ transform: `scale(${scale})`, width: '1220px', height: '520px', position: 'relative' }}
      >
        <div className="round-axis">
          <div className="round-axis__line" />

          {round.nodes.map((node) => {
            const isQ = node.id === 'Q' && showQuestionCard
            const isH = node.id === 'H' && hasReached(hypothesisNode?.status)
            const isE = node.id === 'E' && showFeatureCard
            const isA = node.id === 'A' && showResultCard
            const isU = node.id === 'U' && showResultCard && !hasReached(evaluationNode?.status)

            return (
              <div className="round-col" key={`${round.id}-${node.id}`}>
                {/* Top Branches */}
                {isH && (
                  <div className="branch branch--up">
                    <div className="branch-card">
                      <span className="branch-card__eyebrow">假设分支</span>
                      <span className="branch-card__title">{leadHypothesis?.id ?? '待生成'}</span>
                      <span className="branch-card__text">{leadHypothesis?.label ?? '当前轮尚未生成假设分支。'}</span>
                    </div>
                    <div className="branch-line" />
                  </div>
                )}
                {isE && (
                  <div className="branch branch--up">
                    <div className="branch-card">
                      <span className="branch-card__eyebrow">候选实验</span>
                      <span className="branch-card__title">{leadExperiment?.id ?? '待生成'}</span>
                      <span className="branch-card__text">
                        目标假设 {leadExperiment?.targetHypothesis} · U(E) {leadExperiment?.utility === undefined ? '--' : leadExperiment.utility.toFixed(2)}
                      </span>
                      <div className="branch-card__chips">
                        <span className="branch-card__chip">预期区分度 {leadExperiment?.informationGain?.toFixed(2) ?? '--'}</span>
                        <span className="branch-card__chip">预期性能 {leadExperiment?.performanceGain?.toFixed(2) ?? '--'}</span>
                      </div>
                    </div>
                    <div className="branch-line" />
                  </div>
                )}

                {/* Axis Node */}
                <TimelineNode
                  node={node}
                  isSelected={selectedNodeId === `${round.id}:${node.id}`}
                  onSelect={onSelectNode}
                />

                {/* Bottom Branches */}
                {isQ && (
                  <div className="branch branch--down">
                    <div className="branch-card">
                      <span className="branch-card__eyebrow">问题输入</span>
                      <span className="branch-card__title">{round.questionSummary ? '已输入' : '待输入'}</span>
                      <span className="branch-card__text">
                        {round.questionSummary ? '任务、目标变量与约束已记录。' : '当前尚未输入科学问题。'}
                      </span>
                    </div>
                    <div className="branch-line" />
                  </div>
                )}
                {isA && round.evaluation && (
                  <div className="branch branch--down">
                    <div className="branch-card">
                      <span className="branch-card__eyebrow">评价结果</span>
                      <span className="branch-card__title">
                        {round.evaluation.verdict === 'supports'
                          ? '支持'
                          : round.evaluation.verdict === 'weakens'
                            ? '削弱'
                            : '混合'}
                      </span>
                      <div className="branch-card__chips">
                        <span className="branch-card__chip">delta r {round.evaluation.deltaPearsonR === undefined ? '--' : round.evaluation.deltaPearsonR.toFixed(4)}</span>
                        <span className="branch-card__chip">PG 实际 {round.evaluation.pgActual === undefined ? '--' : round.evaluation.pgActual.toFixed(3)}</span>
                      </div>
                    </div>
                    <div className="branch-line" />
                  </div>
                )}
                {isU && leadUncertainty && (
                  <div className="branch branch--down">
                    <div className="branch-card">
                      <span className="branch-card__eyebrow">不确定性</span>
                      <span className="branch-card__title">{leadUncertainty.id}</span>
                      <span className="branch-card__text">{renderUncertaintyItem(leadUncertainty)}</span>
                    </div>
                    <div className="branch-line" />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
