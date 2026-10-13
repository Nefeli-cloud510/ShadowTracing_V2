import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { reportExportUrl, submitRoundDecision } from '../api/liveWorkflow'
import { DataCoverageAudit } from '../components/DataCoverageAudit'
import {
  ContrastRows,
  LegacyTwoArmGrid,
  RunMetricsGrid,
  summaryMetricText,
} from '../components/ExperimentMetrics'
import { ModelTuningCard } from '../components/ModelTuningCard'
import { PageTabs } from '../components/PageTabs'
import { ThreeLayerConclusionView } from '../components/ThreeLayerConclusionView'
import { useTimelineBundle } from '../hooks/useTimelineBundle'
import { normalizeArmTerms } from '../utils/armTerminology'

const HYPOTHESIS_STATUS_LABELS: Record<string, string> = {
  converged: '可验证',
  active: '活跃',
  observing: '待观察',
  pending: '待定',
  draft: '草稿',
  pruned: '剪枝',
  newly_split: '本轮新增',
  weakened: '已削弱',
}

function hypothesisStatusClass(status?: string): string {
  const normalized = String(status ?? '').toLowerCase()
  return HYPOTHESIS_STATUS_LABELS[normalized] ? normalized : 'pending'
}

function hypothesisStatusLabel(status?: string): string {
  const normalized = String(status ?? '').toLowerCase()
  return HYPOTHESIS_STATUS_LABELS[normalized] ?? '待定'
}

export function RoundReportPage() {
  const navigate = useNavigate()
  const { data, loading, error, refresh } = useTimelineBundle()
  const report = data?.viewModels.roundReport
  const sessionStatus = data?.snapshot.sessionStatus
  const [failureOpen, setFailureOpen] = useState(false)
  const [submittingDecision, setSubmittingDecision] = useState<'continue' | 'adjust' | 'stop' | null>(null)
  const processStage = data?.snapshot.process?.current_stage
  const sessionStage = data?.snapshot.sessionStatus?.stage
  const latestRoundReview = [...(data?.snapshot.decisionLog?.decisions ?? [])]
    .reverse()
    .find((item: any) => item.decision_type === 'round_review_requested')
  const nextRoundSuggestion = report?.threeLayerConclusion?.nextRoundSuggestion
  const overallSummary = report?.threeLayerConclusion?.overallSummary
  const reportSummaryContrast = report?.summary?.contrasts?.[0]
  const deltaRText =
    reportSummaryContrast && reportSummaryContrast.deltaPearsonR != null
    ? summaryMetricText(reportSummaryContrast.deltaPearsonR, 4)
    : summaryMetricText(report?.summary?.deltaPearsonR, 4)
  const deltaRmseText =
    reportSummaryContrast && reportSummaryContrast.deltaRmse != null
    ? summaryMetricText(reportSummaryContrast.deltaRmse, 3)
    : summaryMetricText(report?.summary?.deltaRmse, 3)
  const closureChecklist = Array.isArray(latestRoundReview?.details?.closure_checklist)
    ? latestRoundReview.details.closure_checklist
    : []
  const gatingReady = latestRoundReview?.details?.gating_ready !== false
  const awaitingRoundDecision =
    processStage === 'awaiting_round_decision'
    || sessionStage === 'round_review_requested'
    || sessionStatus?.status === 'awaiting_round_decision'
  const decisionBlockedReason = !awaitingRoundDecision
    ? `当前流程仍在加载中：${data?.viewModels.processMonitor?.currentPhase ?? '闭环阶段未同步'} / ${data?.viewModels.processMonitor?.currentStep ?? '步骤未同步'}。`
    : !gatingReady
      ? `当前仍有环节未完成：${closureChecklist.filter((item: any) => !item.completed).map((item: any) => item.label).join('、') || '请先完成本轮闭环' }。`
      : null
  const failureAnalysis = useMemo(() => {
    const message = sessionStatus?.message ?? ''
    if (!message) {
      return []
    }
    const hints: string[] = []
    if (/目标列|字段|column|KeyError/i.test(message)) {
      hints.push('失败更像是字段映射或数据变量库配置问题，请优先检查目标变量、解释变量和表头白名单是否一致。')
    }
    if (/protocol|协议|feature/i.test(message)) {
      hints.push('失败可能发生在实验协议生成或特征组装阶段，请检查候选实验是否引用了已弃用字段。')
    }
    if (/timeout|连接|api|llm/i.test(message)) {
      hints.push('失败可能包含模型调用或服务链路异常，请同时检查本地服务与外部模型可用性。')
    }
    if (hints.length === 0) {
      hints.push('当前失败由运行态异常触发，请结合执行阶段、字段映射和实验协议继续定位。')
    }
    return hints
  }, [sessionStatus?.message])

  async function handleDecision(decision: 'continue' | 'stop') {
    if (submittingDecision) {
      return
    }
    try {
      setSubmittingDecision(decision)
      const result = await submitRoundDecision({ decision })
      await refresh()
      if (decision === 'stop') {
        navigate('/workflow')
        return
      }
      if (
        result.stage === 'next_round_planning'
        || result.status === 'running'
        || result.planning_status === 'candidate_plan_rebuilt'
      ) {
        navigate('/hypotheses')
      } else {
        navigate('/workflow')
      }
    } finally {
      setSubmittingDecision(null)
    }
  }

  function handleAdjustPlan() {
    if (submittingDecision) {
      return
    }
    navigate('/dialogue')
  }

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <div className="timeline-topbar">
          <PageTabs />
        </div>

        <section className="workspace-page">
          {loading ? (
            <div className="timeline-empty-state">正在读取轮次报告…</div>
          ) : error || !report ? (
            <div className="timeline-empty-state timeline-empty-state--error">
              {error ?? '轮次报告数据不可用。'}
            </div>
          ) : (
            <>
              <header className="workspace-page__header">
                <div>
                  <span className="timeline-header__eyebrow">Round Report</span>
                  <h1>Round {report.roundNumber} 报告与决策</h1>
                  <p>展示性能摘要、科学结论、假设状态与下一步决策建议。</p>
                </div>

                <div className="detail-page__meta">
                  <div className="status-chip">
                    <span>推荐动作</span>
                    <strong>{report.recommendedAction}</strong>
                  </div>
                  <div className="status-chip">
                    <span>Delta r</span>
                    <strong>{deltaRText}</strong>
                  </div>
                  <div className="status-chip">
                    <span>Delta RMSE</span>
                    <strong>{deltaRmseText}</strong>
                  </div>
                  <a
                    className="detail-link detail-link--button"
                    href={reportExportUrl('round', report.roundNumber)}
                    download
                  >
                    导出本轮实验报告
                  </a>
                  <a className="detail-link detail-link--button" href={reportExportUrl('all')} download>
                    导出全部轮次报告
                  </a>
                  {sessionStatus?.status === 'failed' ? (
                    <button type="button" className="detail-link detail-link--button" onClick={() => setFailureOpen(true)}>
                      查看失败原因
                    </button>
                  ) : null}
                </div>
              </header>

              <div className="report-stack">
                <section className="detail-card detail-card--wide">
                  <span className="detail-card__eyebrow">Performance Summary</span>
                  <h2>性能摘要卡片</h2>
                  {report.sourceExperimentId ? (
                    <p className="detail-card__meta">
                      数据来源：实验 {report.sourceExperimentId}（Round {report.roundNumber}）
                      {report.sourceUpdatedAt
                        ? ` · 状态文件更新时间 ${report.sourceUpdatedAt.slice(0, 16).replace('T', ' ')}`
                        : ''}
                    </p>
                  ) : null}
                  <div>
                    <RunMetricsGrid
                      runs={report.summary.runs ?? []}
                      experimentType={report.summary.experimentType}
                    />
                    <ContrastRows contrasts={report.summary.contrasts ?? []} />
                    {(report.summary.runs?.length ?? 0) === 0 ? (
                      <LegacyTwoArmGrid
                        baselinePearsonR={report.summary.baselinePearsonR}
                        treatmentPearsonR={report.summary.treatmentPearsonR}
                        baselineRmse={report.summary.baselineRmse}
                        treatmentRmse={report.summary.treatmentRmse}
                        deltaPearsonR={report.summary.deltaPearsonR}
                        deltaRmse={report.summary.deltaRmse}
                      />
                    ) : null}
                  </div>
                </section>

                <DataCoverageAudit items={report.dataCoverage} />

                <ModelTuningCard
                  modelParameters={report.modelParameters}
                  tuningEntries={report.tuningEntries}
                  tuningNarrative={report.tuningNarrative}
                />

                {report.threeLayerConclusion ? (
                  <ThreeLayerConclusionView conclusion={report.threeLayerConclusion} roundNumber={report.roundNumber} />
                ) : null}

              {report.iterationEvidence ? (
                <section className="detail-card detail-card--wide">
                  <span className="detail-card__eyebrow">Iteration Evidence</span>
                  <h2>给下一轮的建议</h2>
                  {nextRoundSuggestion ? (
                    <div className="next-round-suggestion">
                      {nextRoundSuggestion.evidenceSummary ? (
                        <div className="next-round-suggestion__item">
                          <span>当前证据摘要</span>
                          <p>{nextRoundSuggestion.evidenceSummary}</p>
                        </div>
                      ) : null}
                      <div className="next-round-suggestion__lists">
                        {nextRoundSuggestion.resolvedUncertaintiesThisRound.length > 0 ? (
                          <div className="next-round-suggestion__list next-round-suggestion__list--resolved">
                            <span>本轮已解决的不确定性</span>
                            <ul className="detail-list">
                              {nextRoundSuggestion.resolvedUncertaintiesThisRound.map((item) => (
                                <li key={item}>{item}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        {nextRoundSuggestion.unresolvedUncertaintiesTodo.length > 0 ? (
                          <div className="next-round-suggestion__list next-round-suggestion__list--unresolved">
                            <span>仍需解决的不确定性</span>
                            <ul className="detail-list">
                              {nextRoundSuggestion.unresolvedUncertaintiesTodo.map((item) => (
                                <li key={item}>{item}</li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                      </div>
                      {nextRoundSuggestion.remainingUncertaintyAnalysis ? (
                        <div className="next-round-suggestion__item">
                          <span>剩余不确定性分析</span>
                          <p>{nextRoundSuggestion.remainingUncertaintyAnalysis}</p>
                        </div>
                      ) : null}
                      <div className="next-round-suggestion__grid">
                        {nextRoundSuggestion.hypothesisSpaceAdvice ? (
                          <div className="next-round-suggestion__item">
                            <span>假设空间建议</span>
                            <p>{nextRoundSuggestion.hypothesisSpaceAdvice}</p>
                          </div>
                        ) : null}
                        {nextRoundSuggestion.experimentDesignAdvice ? (
                          <div className="next-round-suggestion__item">
                            <span>实验设计建议</span>
                            <p>{nextRoundSuggestion.experimentDesignAdvice}</p>
                          </div>
                        ) : null}
                        {nextRoundSuggestion.piDecisionAdvice ? (
                          <div className="next-round-suggestion__item">
                            <span>对人工 PI 的决策建议</span>
                            <p>{nextRoundSuggestion.piDecisionAdvice}</p>
                          </div>
                        ) : null}
                      </div>
                      {nextRoundSuggestion.recommendation ? (
                        <div className="next-round-suggestion__recommendation">
                          <span>模型倾向</span>
                          <strong>
                            {nextRoundSuggestion.recommendation === 'continue'
                              ? '进入下一轮'
                              : nextRoundSuggestion.recommendation === 'adjust'
                                ? '调整方向'
                                : '停止实验'}
                          </strong>
                        </div>
                      ) : null}
                      {nextRoundSuggestion.notes.length > 0 ? (
                        <div className="next-round-suggestion__item">
                          <span>注意事项</span>
                          <ul className="detail-list">
                            {nextRoundSuggestion.notes.map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                  <p className="detail-card__meta">
                    数据链：Round {report.iterationEvidence.sourceRound} 实验反馈
                    {report.iterationEvidence.previousExperimentId
                      ? `（${report.iterationEvidence.previousExperimentId}）`
                      : ''}
                    {' → '}
                    Round {report.roundNumber + 1} 候选实验
                    {report.iterationEvidence.nextExperimentId
                      ? `（${report.iterationEvidence.nextExperimentId}）`
                      : ''}
                  </p>

                  {report.candidateEvolution ? (
                    <div className="evaluation-summary-card">
                      <p>{report.candidateEvolution.summary}</p>
                    </div>
                  ) : null}

                  {report.iterationEvidence.previousFocus || report.iterationEvidence.nextFocus ? (
                    <div className="collection-grid">
                      {report.iterationEvidence.previousFocus ? (
                        <article className="collection-card">
                          <strong>上一轮实验焦点</strong>
                          <p>{report.iterationEvidence.previousFocus}</p>
                          <div className="collection-card__meta">
                            <span>{report.iterationEvidence.previousExperimentId ?? '--'}</span>
                            <span>Round {report.iterationEvidence.sourceRound}</span>
                          </div>
                        </article>
                      ) : null}
                      {report.iterationEvidence.nextFocus ? (
                        <article className="collection-card">
                          <strong>本轮候选焦点</strong>
                          <p>{report.iterationEvidence.nextFocus}</p>
                          <div className="collection-card__meta">
                            <span>{report.iterationEvidence.nextExperimentId ?? '--'}</span>
                            <span>Round {report.roundNumber + 1}</span>
                          </div>
                        </article>
                      ) : null}
                    </div>
                  ) : null}

                  {report.iterationEvidence.validations.length > 0 ? (
                    <>
                      <h3>迭代生效校验</h3>
                      <div className="decision-option-row decision-option-row--wrap">
                        {report.iterationEvidence.validations.map((item) => (
                          <div key={item.itemId} className="status-chip">
                            <span>{item.label}</span>
                            <strong>{item.passed ? '已通过' : '未通过'}</strong>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : null}

                  {report.iterationEvidence.inputSources.length > 0 ? (
                    <>
                      <h3>本轮输入来源</h3>
                      <p className="detail-card__meta">
                        来源文件：round_history.json · 字段 iteration_input_sources
                      </p>
                      <ul className="detail-list">
                        {report.iterationEvidence.inputSources.slice(0, 8).map((source) => {
                          const [type, ...rest] = source.split(':')
                          const label =
                            { round_archive: '轮次归档', tested_hypothesis: '已检验假设', target_uncertainty: '目标不确定性', scientific_finding: '科学发现', metric_delta: '指标变化', reasoning_trace: '推理轨迹' }[type] ?? type
                          return (
                            <li key={source}>
                              <strong>{label}</strong>
                              {rest.length > 0 ? `：${normalizeArmTerms(rest.join(':'))}` : ''}
                            </li>
                          )
                        })}
                      </ul>
                      {report.iterationEvidence.inputSources.length > 8
                        ? (
                            <p className="detail-card__meta">
                              另有 {report.iterationEvidence.inputSources.length - 8} 条来源记录，可在状态文件中查看完整列表。
                            </p>
                          )
                        : null}
                    </>
                  ) : null}
                </section>
              ) : null}
              </div>

              <div className="workspace-page__grid">
                <section className="detail-card">
                  <span className="detail-card__eyebrow">Scientific Conclusions</span>
                  <h2>科学结论</h2>
                  {report.sourceExperimentId ? (
                    <p className="detail-card__meta">
                      数据来源：实验 {report.sourceExperimentId}（Round {report.roundNumber}）
                      {report.sourceUpdatedAt
                        ? ` · 状态文件更新时间 ${report.sourceUpdatedAt.slice(0, 16).replace('T', ' ')}`
                        : ''}
                    </p>
                  ) : null}
                  {overallSummary ? (
                    <p className="scientific-overall-summary">{overallSummary}</p>
                  ) : (
                    <ul className="detail-list">
                      {report.scientificConclusions.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  )}
                </section>

                <section className="detail-card">
                  <span className="detail-card__eyebrow">Hypothesis State</span>
                  <h2>假设空间状态</h2>
                  <div className="collection-grid">
                    {report.highlightedHypotheses.map((item, index) => (
                      <article
                        key={item.id}
                        className={`collection-card hypothesis-state-card hypothesis-state-card--${hypothesisStatusClass(item.status)}`}
                      >
                        <strong>{item.displayLabel ?? `假设 ${index + 1}`}</strong>
                        <p>{item.label}</p>
                        <div className="collection-card__meta">
                          <span className={`hypothesis-status-badge hypothesis-status-badge--${hypothesisStatusClass(item.status)}`}>
                            {hypothesisStatusLabel(item.status)}
                          </span>
                          <span>支持度 {item.supportScore.toFixed(3)}</span>
                        </div>
                      </article>
                    ))}
                  </div>
                </section>

                <section className="detail-card detail-card--wide">
                  <span className="detail-card__eyebrow">Unresolved Questions</span>
                  <h2>剩余不确定性</h2>
                  <ul className="detail-list">
                    {report.unresolvedQuestions.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>

                <section className="detail-card detail-card--wide">
                  <span className="detail-card__eyebrow">Decision</span>
                  <h2>轮次迭代入口</h2>
                  <div className="decision-option-row">
                    <button
                      type="button"
                      className="detail-link detail-link--button detail-link--accent"
                      onClick={() => void handleDecision('continue')}
                      disabled={submittingDecision !== null}
                    >
                      {submittingDecision === 'continue' ? '加载中…' : '开启新一轮迭代'}
                    </button>
                    <button
                      type="button"
                      className="detail-link detail-link--button"
                      onClick={handleAdjustPlan}
                      disabled={submittingDecision !== null}
                    >
                      调整计划
                    </button>
                    <button
                      type="button"
                      className="detail-link detail-link--button"
                      onClick={() => void handleDecision('stop')}
                      disabled={submittingDecision !== null}
                    >
                      {submittingDecision === 'stop' ? '加载中…' : '停止实验'}
                    </button>
                  </div>
                  {decisionBlockedReason ? <p>当前状态提示：{decisionBlockedReason} 但你仍可从此处强制推进新一轮。</p> : null}
                  {submittingDecision ? (
                    <p>加载中：系统正在汇总当前科学解释、假设空间状态与轮次日志，并同步下一步页面。生成大约需要5-15分钟，请耐心等待。</p>
                  ) : null}
                  {!gatingReady && closureChecklist.length > 0 ? (
                    <ul className="detail-list">
                      {closureChecklist
                        .filter((item: any) => !item.completed)
                        .map((item: any) => (
                          <li key={item.item_id}>{item.label}：{item.detail ?? '尚未完成'}</li>
                        ))}
                    </ul>
                  ) : null}
                </section>
              </div>
            </>
          )}
        </section>
        {failureOpen ? (
          <div className="preview-lightbox" role="dialog" aria-modal="true">
            <div className="preview-lightbox__panel preview-lightbox__panel--report">
              <div className="preview-lightbox__header">
                <h2>失败原因解析</h2>
                <button type="button" className="detail-link detail-link--button" onClick={() => setFailureOpen(false)}>
                  关闭
                </button>
              </div>
              <div className="detail-card detail-card--wide">
                <span className="detail-card__eyebrow">Failure Report</span>
                <h2>失败摘要</h2>
                <p>{sessionStatus?.message ?? '当前没有可用的失败信息。'}</p>
                <ul className="detail-list">
                  {failureAnalysis.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  )
}
