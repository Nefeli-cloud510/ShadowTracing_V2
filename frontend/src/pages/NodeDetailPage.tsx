import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { submitApprovalAction } from '../api/liveWorkflow'
import { ApprovalOverlay } from '../components/ApprovalOverlay'
import { CandidateExperimentTable } from '../components/CandidateExperimentTable'
import { PageTabs } from '../components/PageTabs'
import { UncertaintyQueueList } from '../components/UncertaintyQueueList'
import { useTimelineBundle } from '../hooks/useTimelineBundle'
import { filterSelectableCandidateExperiments } from '../utils/experimentValidation'
import { useTabs } from '../contexts/TabContext'

function formatCodeLabel(value?: string, fallback = '待更新') {
  if (!value) {
    return fallback
  }

  const labels: Record<string, string> = {
    active: '活跃',
    observing: '待观察',
    converged: '可验证',
    draft: '草稿',
    pruned: '剪枝',
    supported: '支持',
    partially_supported: '部分支持',
    weakened: '已削弱',
    resolved: '已解决',
    tracking: '持续跟踪',
    pending: '待执行',
    completed: '已完成',
    awaiting_human_approval: '等待 PI 审批',
    awaiting_hypothesis_confirmation: '等待假设树确认',
    hypothesis_tree_confirmed: '假设树已确认',
    experiment_selection_requested: '提交实验审批',
    experiment_approved: '实验已批准',
    round_review_requested: '轮次复盘申请',
  }

  return labels[value] ?? value.replace(/_/g, ' ')
}

function hypothesisNodeStatusLabel(status?: string, fallback = '待更新') {
  const labels: Record<string, string> = {
    active: '活跃',
    observing: '待观察',
    converged: '可验证',
    draft: '草稿',
    pending: '待定',
    pruned: '剪枝',
  }
  return labels[status ?? ''] ?? formatCodeLabel(status, fallback)
}

function buildDisplayMapper(rawPlannerInput?: any) {
  const rawMap = rawPlannerInput?.data_dictionary_summary?.raw_display_map ?? {}
  const names = Object.keys(rawMap).filter((key) => rawMap[key] && rawMap[key] !== key)
  return (text?: string) => {
    if (!text) {
      return text ?? ''
    }
    return [...names].sort((a, b) => b.length - a.length).reduce((result, raw) => result.split(raw).join(rawMap[raw] ?? raw), String(text))
  }
}

function statementLabel(tree?: any, ids: string[] = [], displayText?: (text?: string) => string): string {
  const nodeById = new Map<string, string>()
  for (const node of tree?.nodes ?? []) {
    if (node.hypothesis_id && node.statement) {
      nodeById.set(node.hypothesis_id, node.statement)
    }
  }
  const labels = ids.slice(0, 2).map((id) => {
    const statement = nodeById.get(id)
    return displayText?.(statement ?? id) || statement || id
  })
  return labels.join('；') || '本轮以不确定性验证为主'
}

function uncertaintyLabel(
  uncertainties?: any,
  id?: string,
  displayText?: (text?: string) => string,
): string {
  const record = (uncertainties?.records ?? []).find((item: any) => item.uncertainty_id === id)
  const question = record?.question ?? record?.description
  return displayText?.(question) || question || id || '未命名不确定性'
}

function summarizeArtifact(path?: string) {
  if (!path) {
    return '未生成'
  }
  return path.replace(/\\/g, '/').split('/').pop() ?? path
}

function QuestionDetail({ rawTask, displayText }: { rawTask: any; displayText: (text?: string) => string }) {
  if (!rawTask || !rawTask.payload?.research_question) return <p>暂无科学问题数据。</p>
  const q = rawTask.payload.research_question
  return (
    <div className="detail-panel">
      <div className="detail-panel__section">
        <h3>核心科学问题</h3>
        <p>{q.text}</p>
      </div>
      <div className="detail-panel__section">
        <h3>变量配置</h3>
        <ul>
          <li><strong>研究目标：</strong> {displayText(q.target || q.variables?.y) || '待识别'}</li>
          <li><strong>主要解释信号：</strong> {displayText(q.variables?.x) || '待识别'}</li>
          <li><strong>候选特征：</strong> {(q.variables?.m_candidates ?? []).map(displayText).join(', ') || '暂无'}</li>
        </ul>
      </div>
      {rawTask.payload.constraints && (
        <div className="detail-panel__section">
          <h3>实验约束</h3>
          <ul>
            <li>严格避免未来信息泄露：{rawTask.payload.constraints.no_future_information ? '是' : '否'}</li>
            <li>最大闭环轮数：{rawTask.payload.constraints.max_rounds ?? '--'}</li>
          </ul>
        </div>
      )}
    </div>
  )
}

function KnowledgeDetail({ rawPlannerInput, displayText }: { rawPlannerInput: any; displayText: (text?: string) => string }) {
  if (!rawPlannerInput) return <p>暂无知识注入数据。</p>
  const guidance =
    rawPlannerInput.planner_guidance?.filter(
      (g: string) => g.startsWith('rag_') || g.includes('文献') || g.includes('LHAASO'),
    ) ?? []

  return (
    <div className="detail-panel">
      <div className="detail-panel__section">
        <h3>知识线索</h3>
        <ul className="rag-list">
          {guidance.length > 0 ? guidance.map((item: string, idx: number) => (
            <li key={idx} className="rag-list__item">{item}</li>
          )) : <li className="rag-list__item">当前轮暂无显式 RAG 文献片段，已回退为任务与记忆上下文。</li>}
        </ul>
      </div>
      {rawPlannerInput.data_dictionary_summary ? (
        <div className="detail-panel__section">
          <h3>数据变量库摘要</h3>
          <ul>
            <li><strong>数据集：</strong> {rawPlannerInput.data_dictionary_summary.dataset_name}</li>
            <li><strong>时间对齐字段：</strong> {displayText(rawPlannerInput.data_dictionary_summary.display_time_column ?? rawPlannerInput.data_dictionary_summary.time_column)}</li>
            <li><strong>候选特征：</strong> {(rawPlannerInput.data_dictionary_summary.display_feature_candidates ?? rawPlannerInput.data_dictionary_summary.feature_candidates ?? []).map(displayText).join(', ')}</li>
          </ul>
        </div>
      ) : null}
    </div>
  )
}

function ScientificQuestioningDetail({ rawPlannerInput, displayText }: { rawPlannerInput: any; displayText: (text?: string) => string }) {
  if (!rawPlannerInput || !rawPlannerInput.recent_reasoning_traces) return <p>暂无科学质询记录。</p>
  const traces = rawPlannerInput.recent_reasoning_traces.filter((t: any) => t.stage === 'scientific_questioner')
  
  if (traces.length === 0) return <p>当前轮无质询记录。</p>
  
  return (
    <div className="detail-panel">
      {traces.map((t: any, idx: number) => (
        <div key={idx} className="trace-card">
          <div className="trace-card__header">
            <strong>质询记录 {idx + 1}</strong>
          </div>
          <p className="trace-card__summary">{displayText(t.summary)}</p>
        </div>
      ))}
    </div>
  )
}

function HypothesisTreeDetail({ rawTree, displayText }: { rawTree: any; displayText: (text?: string) => string }) {
  if (!rawTree || !rawTree.nodes) return <p>暂无假设树数据。</p>
  return (
    <div className="tree-container">
      {(Array.isArray(rawTree.nodes) ? rawTree.nodes : Object.values(rawTree.nodes)).map((n: any, nodeIndex: number) => {
        const nodeId = n.hypothesis_id ?? n.id ?? 'H_unknown'
        const rawStatement = n.statement ?? n.label ?? ''
        const nodeLabel = displayText(rawStatement) || `假设 ${nodeIndex + 1}`
        const depth = Math.max(0, Number(n.level ?? 1) - 1)
        const indent = Math.max(0, depth * 40);
        return (
          <div key={nodeId} className="tree-node" style={{ marginLeft: `${indent}px` }}>
            <div className={`tree-node__card tree-node__card--${n.status}`}>
              <strong>{nodeIndex === 0 ? '主假设' : `假设 ${nodeIndex + 1}`}</strong>
              <span className="tree-node__statement">{nodeLabel}</span>
              <div className="tree-node__meta">
                <span>支持度: {Number(n.support_score || 0).toFixed(2)}</span>
                <span>状态: {hypothesisNodeStatusLabel(n.status, '状态待更新')}</span>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function CandidateExperimentDetail({
  rawExperiments,
  rawTree,
  rawUncertainties,
  displayText,
}: {
  rawExperiments: any
  rawTree: any
  rawUncertainties: any
  displayText: (text?: string) => string
}) {
  const candidates = filterSelectableCandidateExperiments(rawExperiments?.candidates ?? [])
  if (candidates.length === 0) return <p>暂无可审批的区分性候选实验数据。</p>
  return (
    <CandidateExperimentTable
      candidates={candidates}
      treeNodes={rawTree?.nodes ?? []}
      uncertaintyRecords={rawUncertainties?.records ?? []}
      displayText={displayText}
    />
  )
}

function ApprovalDetail({ rawLog, rawTree, displayText }: { rawLog: any; rawTree: any; displayText: (text?: string) => string }) {
  if (!rawLog || !rawLog.decisions) return <p>暂无审批记录。</p>
  const requestByRound = new Map<number, any>()
  const approvalByRound = new Map<number, any>()
  const reviewRequests = rawLog.decisions.filter((d: any) => d.decision_type === 'round_review_requested')
  rawLog.decisions.forEach((decision: any) => {
    if (decision.decision_type === 'experiment_selection_requested') {
      requestByRound.set(decision.round_id ?? 0, decision)
    }
    if (decision.decision_type === 'experiment_approved') {
      approvalByRound.set(decision.round_id ?? 0, decision)
    }
  })
  const approvals = rawLog.decisions.filter(
    (d: any) => d.made_by === 'human_pi' || d.decision_type?.includes('approved') || d.decision_type?.includes('round_'),
  )
  return (
    <div className="approval-container">
      {reviewRequests.map((request: any) => {
        const requested = requestByRound.get(request.round_id ?? 0)
        const approved = approvalByRound.get(request.round_id ?? 0)
        return (
          <div key={request.decision_id} className="approval-card approval-card--compare">
          <div className="approval-card__header">
            <strong>{request.summary ?? request.decision_type}</strong>
            <span>Round {request.round_id ?? '--'}</span>
          </div>
          <div className="approval-side-by-side approval-side-by-side--compact">
            <section className="approval-side-by-side__panel">
              <span className="approval-compare-card__label">候选实验</span>
              <strong>{requested?.details?.candidate_id ?? '--'}</strong>
              <p>{requested?.summary ?? '暂无候选实验请求'}</p>
              <div className="approval-metric-grid">
                <span>预期区分度 {Number(requested?.details?.information_gain ?? 0).toFixed(3)}</span>
                <span>PG {Number(requested?.details?.performance_gain ?? 0).toFixed(3)}</span>
                <span>Risk {Number(requested?.details?.risk ?? 0).toFixed(3)}</span>
                <span>Cost {Number(requested?.details?.cost ?? 0).toFixed(3)}</span>
                <span>U(E) {Number(requested?.details?.utility_score ?? 0).toFixed(3)}</span>
              </div>
            </section>
            <section className="approval-side-by-side__panel approval-side-by-side__panel--approved">
              <span className="approval-compare-card__label">最终批准</span>
              <strong>{approved?.details?.candidate_id ?? '--'}</strong>
              <p>{approved?.summary ?? '当前尚未批准'}</p>
              <div className="approval-metric-grid">
                <span>预期区分度 {Number(approved?.details?.information_gain ?? 0).toFixed(3)}</span>
                <span>PG {Number(approved?.details?.performance_gain ?? 0).toFixed(3)}</span>
                <span>Risk {Number(approved?.details?.risk ?? 0).toFixed(3)}</span>
                <span>Cost {Number(approved?.details?.cost ?? 0).toFixed(3)}</span>
                <span>U(E) {Number(approved?.details?.utility_score ?? 0).toFixed(3)}</span>
              </div>
            </section>
          </div>
          <p>实验: {request.details?.experiment_id ?? '--'}</p>
          {request.details?.evaluation_summary ? (
            <div className="approval-compare-mini">
              <div className="approval-compare-mini__metrics">
                <span>对照组相关性 {Number(request.details.evaluation_summary.baseline_pearson_r ?? 0).toFixed(4)}</span>
                <span>实验组相关性 {Number(request.details.evaluation_summary.treatment_pearson_r ?? 0).toFixed(4)}</span>
                <span>改善幅度 {Number(request.details.evaluation_summary.delta_pearson_r ?? 0).toFixed(4)}</span>
              </div>
              <ul className="detail-list">
                {(request.details.hypothesis_assessments ?? []).slice(0, 4).map((item: any) => (
                  <li key={item.hypothesis_id}>
                    {statementLabel(rawTree, [item.hypothesis_id], displayText)}：支持度从 {Number(item.support_before ?? 0).toFixed(3)} 调整到 {Number(item.support_after ?? 0).toFixed(3)}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          </div>
        )
      })}
      {approvals.map((a: any, idx: number) => (
        <div key={idx} className="approval-card">
          <div className="approval-card__header">
            <strong>{a.summary ?? a.decision_type}</strong>
            <span>{new Date(a.timestamp).toLocaleString()}</span>
          </div>
          <p>动作类型：{formatCodeLabel(a.decision_type, '审批记录')}</p>
          {a.details?.candidate_id ? <p>候选实验：{a.details.candidate_id}</p> : null}
          {a.details?.notes ? <p>备注：{a.details.notes}</p> : null}
          {a.details?.human_feedback && (
            <div className="approval-card__feedback">
              <strong>PI 反馈:</strong> {a.details.human_feedback}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function UncertaintyDetail({
  rawUncertainties,
  rawTree,
  displayText,
}: {
  rawUncertainties: any
  rawTree: any
  displayText: (text?: string) => string
}) {
  if (!rawUncertainties || !rawUncertainties.records) return <p>暂无不确定性数据。</p>
  const activeRecords = rawUncertainties.records.filter((r: any) => r.status === 'active')
  const nodeById = new Map<string, any>()
  for (const node of rawTree?.nodes ?? []) {
    if (node.hypothesis_id) {
      nodeById.set(node.hypothesis_id, node)
    }
  }
  const hypothesisLabel = (id?: string) => {
    if (!id) {
      return '--'
    }
    const node = nodeById.get(id)
    return node?.display_hypothesis_id || (node?.level ? `H${node.level}` : id)
  }
  return (
    <UncertaintyQueueList records={activeRecords} displayText={displayText} hypothesisLabel={hypothesisLabel} />
  )
}

function EvaluationDetail({ rawPlannerInput }: { rawPlannerInput: any }) {
  if (!rawPlannerInput || !rawPlannerInput.evaluation_summary) return <p>当前轮暂无评价结果。</p>
  const summary = rawPlannerInput.evaluation_summary
  
  return (
    <div className="detail-panel">
      <div className="detail-panel__section">
        <h3>实验指标对比</h3>
        <div className="metrics-grid">
          <div className="metric-box">
            <span className="metric-box__label">对照组相关性</span>
            <span className="metric-box__value">{Number(summary.baseline_pearson_r || 0).toFixed(4)}</span>
          </div>
          <div className="metric-box">
            <span className="metric-box__label">实验组相关性</span>
            <span className="metric-box__value">{Number(summary.treatment_pearson_r || 0).toFixed(4)}</span>
          </div>
          <div className="metric-box metric-box--highlight">
            <span className="metric-box__label">改善幅度</span>
            <span className="metric-box__value">{Number(summary.delta_pearson_r || 0).toFixed(4)}</span>
          </div>
        </div>
      </div>
      <div className="detail-panel__section">
        <h3>稳健性评估</h3>
        <p>{summary.robustness_recommendation}</p>
        <span className={`badge ${summary.stable ? 'badge--success' : 'badge--warning'}`}>
          {summary.stable ? '稳定' : '待复核'}
        </span>
      </div>
    </div>
  )
}

function ExperimentExecutionDetail({ rawExperimentMemory }: { rawExperimentMemory: any }) {
  const latestEntry = [...(rawExperimentMemory?.entries ?? [])].sort(
    (a: any, b: any) => (b.round_id ?? 0) - (a.round_id ?? 0),
  )[0]

  if (!latestEntry) return <p>当前轮暂无实验执行记录。</p>

  return (
    <div className="detail-panel">
      <div className="detail-panel__section">
        <h3>执行协议</h3>
        <ul>
          <li><strong>实验编号：</strong> {latestEntry.experiment_id}</li>
          <li><strong>执行状态：</strong> {formatCodeLabel(latestEntry.status)}</li>
          <li><strong>协议产物：</strong> {latestEntry.protocol_path ? '已生成' : '未生成'}</li>
          <li><strong>结果归档：</strong> {latestEntry.result_path ? '已完成' : '未完成'}</li>
          <li><strong>评价报告：</strong> {latestEntry.evaluation_path ? '已生成' : '未生成'}</li>
        </ul>
      </div>
      {latestEntry.key_findings?.length ? (
        <div className="detail-panel__section">
          <h3>关键发现</h3>
          <ul className="rag-list">
            {latestEntry.key_findings.map((item: string) => (
              <li key={item} className="rag-list__item">{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {latestEntry.visualizations?.length ? (
        <div className="detail-panel__section">
          <h3>已生成图表产物</h3>
          <ul className="rag-list">
            {latestEntry.visualizations.slice(0, 6).map((item: string) => (
              <li key={item} className="rag-list__item">{summarizeArtifact(item)}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}

function WritebackDetail({
  rawPlannerInput,
  rawProcess,
  rawTree,
  rawUncertainties,
  displayText,
}: {
  rawPlannerInput: any
  rawProcess: any
  rawTree: any
  rawUncertainties: any
  displayText: (text?: string) => string
}) {
  if (!rawPlannerInput) return <p>暂无状态回写记录。</p>
  
  return (
    <div className="detail-panel">
      <div className="detail-panel__section">
        <h3>轮次推进状态</h3>
        <ul>
          <li><strong>当前轮：</strong> {rawProcess?.current_round ?? '--'}</li>
          <li><strong>下一轮：</strong> {rawPlannerInput.next_round_id ?? '未生成'}</li>
          <li><strong>当前阶段：</strong> {formatCodeLabel(rawProcess?.current_phase, '待更新')}</li>
        </ul>
      </div>
      {rawPlannerInput.recent_hypothesis_assessments && (
        <div className="detail-panel__section">
          <h3>假设树支持度更新</h3>
          <ul className="rag-list">
            {rawPlannerInput.recent_hypothesis_assessments.map((h: any, idx: number) => (
              <li key={idx} className="rag-list__item">
                <strong>{statementLabel(rawTree, [h.hypothesis_id], displayText)}：</strong> 支持度从 {h.support_before.toFixed(2)} 调整到 {h.support_after.toFixed(2)}，当前状态为 {formatCodeLabel(h.status)}
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {rawPlannerInput.recent_disagreement_updates && (
        <div className="detail-panel__section">
          <h3>不确定性消解更新</h3>
          <ul className="rag-list">
            {rawPlannerInput.recent_disagreement_updates.map((u: any, idx: number) => (
              <li key={idx} className="rag-list__item">
                <strong>{uncertaintyLabel(rawUncertainties, u.uncertainty_id, displayText)}：</strong> 分歧跨度从 {u.support_span_before.toFixed(3)} 调整到 {u.support_span_after.toFixed(3)}，当前状态为 {formatCodeLabel(u.resolution_status)}
              </li>
            ))}
          </ul>
        </div>
      )}
      {rawPlannerInput.recent_human_feedback?.length ? (
        <div className="detail-panel__section">
          <h3>最近人工反馈</h3>
          <ul className="rag-list">
            {rawPlannerInput.recent_human_feedback.map((item: any, idx: number) => (
              <li key={idx} className="rag-list__item">{item.content ?? item.summary ?? '人工反馈已记录'}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}

export function NodeDetailPage({ roundId, nodeId }: { roundId: string, nodeId: string }) {
  const navigate = useNavigate()
  const { data, loading, error, refresh } = useTimelineBundle()
  const { removeTab, setActiveTabId } = useTabs()
  const [approvalOpen, setApprovalOpen] = useState(false)

  const round = useMemo(() => {
    if (!data) return null
    return data.viewModels.rounds.find((item) => item.id === roundId) ?? null
  }, [data, roundId])

  const node = round?.nodes.find((item) => item.id === nodeId) ?? round?.nodes[0] ?? null
  const snapshot = data?.snapshot
  const displayText = useMemo(() => buildDisplayMapper(snapshot?.plannerInput), [snapshot?.plannerInput])

  function goToRelatedPage() {
    if (node?.relatedPage) {
      setActiveTabId(node.relatedPage)
    }
  }

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <div className="timeline-topbar">
          <PageTabs />
        </div>

        <section className="detail-page">
          {loading ? (
            <div className="timeline-empty-state">正在同步当前节点状态…</div>
          ) : error ? (
            <div className="timeline-empty-state timeline-empty-state--error">{error}</div>
          ) : round && node ? (
            <>
              <header className="detail-page__header">
                <div>
                  <span className="timeline-header__eyebrow">节点详情页</span>
                  <h1>{node.title}</h1>
                  <p>{node.summary}</p>
                </div>

                <div className="detail-page__meta">
                  <div className="status-chip">
                    <span>Round</span>
                    <strong>{round.title}</strong>
                  </div>
                  <div className="status-chip">
                    <span>Node</span>
                    <strong>{node.id}</strong>
                  </div>
                  <div className="status-chip status-chip--outline">
                    <span>Status</span>
                    <strong>{node.status}</strong>
                  </div>
                </div>
              </header>

              <div className="detail-page__grid">
                {node.id === 'Q' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">Task Definition</span>
                    <h2>科学问题与任务约束</h2>
                    <QuestionDetail rawTask={snapshot?.task} displayText={displayText} />
                  </section>
                )}

                {node.id === 'K' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">Knowledge Injection</span>
                    <h2>RAG 知识库检索</h2>
                    <KnowledgeDetail rawPlannerInput={snapshot?.plannerInput} displayText={displayText} />
                  </section>
                )}

                {node.id === 'C' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">Scientific Questioning</span>
                    <h2>科学质询与分歧识别</h2>
                    <ScientificQuestioningDetail rawPlannerInput={snapshot?.plannerInput} displayText={displayText} />
                  </section>
                )}

                {node.id === 'H' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">Hypothesis Tree</span>
                    <h2>科学假设树</h2>
                    <HypothesisTreeDetail rawTree={snapshot?.hypothesisTree} displayText={displayText} />
                  </section>
                )}

                {node.id === 'U' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">Uncertainties</span>
                    <h2>不确定性队列</h2>
                    <UncertaintyDetail
                      rawUncertainties={snapshot?.uncertainties}
                      rawTree={snapshot?.hypothesisTree}
                      displayText={displayText}
                    />
                  </section>
                )}
                
                {node.id === 'E' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">Candidate Matrix</span>
                    <h2>候选实验矩阵</h2>
                    <CandidateExperimentDetail
                      rawExperiments={snapshot?.candidateExperiments}
                      rawTree={snapshot?.hypothesisTree}
                      rawUncertainties={snapshot?.uncertainties}
                      displayText={displayText}
                    />
                  </section>
                )}

                {node.id === 'P' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">PI 审批</span>
                    <h2>人工审批记录</h2>
                    <ApprovalDetail rawLog={snapshot?.decisionLog} rawTree={snapshot?.hypothesisTree} displayText={displayText} />
                  </section>
                )}

                {node.id === 'X' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">实验执行</span>
                    <h2>协议、结果与产物</h2>
                    <ExperimentExecutionDetail rawExperimentMemory={snapshot?.experimentMemory} />
                  </section>
                )}

                {node.id === 'A' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">实验指标与对照解释</span>
                    <h2>对照组/实验组指标与推理对照</h2>
                    <EvaluationDetail rawPlannerInput={snapshot?.plannerInput} />
                  </section>
                )}

                {node.id === 'W' && (
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">状态回写</span>
                    <h2>状态回写影响</h2>
                    <WritebackDetail
                      rawPlannerInput={snapshot?.plannerInput}
                      rawProcess={snapshot?.process}
                      rawTree={snapshot?.hypothesisTree}
                      rawUncertainties={snapshot?.uncertainties}
                      displayText={displayText}
                    />
                  </section>
                )}

                <section className="detail-card">
                  <span className="detail-card__eyebrow">Round Summary</span>
                  <h2>当前轮摘要</h2>
                  <ul className="detail-list">
                    {(node.detailItems ?? []).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>

                <section className="detail-card">
                  <span className="detail-card__eyebrow">Related Workspace</span>
                  <h2>关联页面</h2>
                  <ul className="detail-list">
                    <li>优先从主页面查看当前轮闭环位置。</li>
                    <li>深入数据请跳转到该节点关联的一级页面。</li>
                    <li>节点详情只保留局部深挖，不再堆叠全局信息。</li>
                  </ul>
                </section>

                <section className="detail-card detail-card--wide">
                  <span className="detail-card__eyebrow">下一步入口</span>
                  <h2>相关操作</h2>
                  <div className="detail-links">
                    {node.relatedPage ? (
                      <button
                        type="button"
                        className="detail-link detail-link--button"
                        onClick={goToRelatedPage}
                      >
                        打开关联页面
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="detail-link detail-link--button"
                      onClick={() => removeTab(`node-${roundId}-${nodeId}`)}
                    >
                      关闭当前页面
                    </button>
                    {node.id === 'P' && data ? (
                      <button
                        type="button"
                        className="detail-link detail-link--button"
                        onClick={() => setApprovalOpen(true)}
                      >
                        打开审批弹窗
                      </button>
                    ) : null}
                  </div>
                </section>
              </div>
            </>
          ) : (
            <div className="timeline-empty-state">暂无详情数据。</div>
          )}
        </section>
        {node?.id === 'P' && data ? (
          <ApprovalOverlay
            data={data.viewModels.approvalOverlay}
            open={approvalOpen}
            onClose={() => setApprovalOpen(false)}
            onAction={async (action, payload) => {
              const result = await submitApprovalAction({
                action,
                candidateId: payload.candidateId,
                humanNotes: payload.humanNotes,
              })
              setApprovalOpen(false)
              await refresh()
              if (action === 'approve' || action === 'modify') {
                setActiveTabId('execution')
                navigate('/execution')
              } else if (result.stage === 'approval_pending' || result.status === 'awaiting_approval') {
                setActiveTabId('approval')
                navigate('/approval')
              }
            }}
          />
        ) : null}
      </div>
    </div>
  )
}
