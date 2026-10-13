import { useMemo, useState } from 'react'
import { resetWorkflow } from '../api/liveWorkflow'
import { PageTabs } from '../components/PageTabs'
import {
  ContrastRows,
  DiagnosticsList,
  RunMetricsGrid,
  experimentTypeLabel,
} from '../components/ExperimentMetrics'
import { useTimelineBundle } from '../hooks/useTimelineBundle'
import { useTabs } from '../contexts/TabContext'
import { resetClientWorkspaceState } from '../utils/missionControllerPersistence'

function hypothesisStatusLabel(status: string) {
  const mapping: Record<string, string> = {
    active: '活跃',
    newly_split: '本轮新增',
    observing: '待观察',
    converged: '可验证',
    draft: '草稿',
    pending: '待定',
    pruned: '剪枝',
    weakened: '已削弱',
  }
  return mapping[status] ?? status
}

export function ClosedLoopTimelinePage() {
  const { data, loading, error, refreshing, refresh } = useTimelineBundle()
  const rounds = data?.viewModels.rounds ?? []
  const mission = data?.viewModels.mission
  const processMonitor = data?.viewModels.processMonitor
  const knowledgeBase = data?.viewModels.knowledgeBaseManager
  const sessionStatus = data?.snapshot.sessionStatus
  const { addTab } = useTabs()
  const [showHistoryRounds, setShowHistoryRounds] = useState(false)
  const [showHypothesisTree, setShowHypothesisTree] = useState(false)
  const [resetting, setResetting] = useState(false)
  const currentRound = useMemo(() => rounds.find((round) => round.isCurrent) ?? rounds[rounds.length - 1] ?? null, [rounds])
  const displayQuestion = mission?.scientificQuestion || currentRound?.questionSummary || ''
  const knowledgeCount = knowledgeBase?.collections.reduce((sum, item) => sum + item.count, 0) ?? 0
  const knowledgeUploadCount = knowledgeBase?.uploads.length ?? 0
  const dataCount = mission?.dataSources.length ?? 0
  const dataTableNames = useMemo(
    () =>
      mission?.dataSources
        .map((item) => item.path?.replace(/\\/g, '/').split('/').pop() ?? item.id)
        .filter((item, index, array) => Boolean(item) && array.indexOf(item) === index) ?? [],
    [mission?.dataSources],
  )

  const displayedRounds = useMemo(() => {
    if (showHistoryRounds) {
      return rounds
    }
    return rounds.filter((round) => round.isCurrent).slice(-1)
  }, [rounds, showHistoryRounds])

  function handleOpenNode(roundId: string, nodeId: string, shortLabel: string) {
    addTab({
      id: `node-${roundId}-${nodeId}`,
      title: `${shortLabel} 详情`,
      path: `/node/${roundId}/${nodeId}`,
      roundId,
      nodeId,
    })
  }

  async function handleClearMemory() {
    try {
      setResetting(true)
      try {
        await resetWorkflow()
      } catch {
        // Frontend state should still be cleared even if the live server is unavailable.
      }
      await resetClientWorkspaceState()
      await refresh()
    } finally {
      setResetting(false)
    }
  }

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame">
        <div className="timeline-topbar">
          <PageTabs />
        </div>

        <header className="timeline-header">
          <div className="timeline-header__brand">
            <span className="timeline-header__eyebrow">ShadowTracing</span>
            <h1>逐影ShadowTracing</h1>
            <p>竞争假设驱动的实验规划与反馈系统</p>
          </div>

          <div className="timeline-nav timeline-nav--tools">
            <div className="status-chip">
              <span>运行模型</span>
              <strong>{String(sessionStatus?.model ?? 'qwen3.8-flash')}</strong>
            </div>
            <button
              type="button"
              className="timeline-nav__pill"
              onClick={() => setShowHypothesisTree((value) => !value)}
            >
              {showHypothesisTree ? '折叠假设树' : '展开假设树'}
            </button>
            <button
              type="button"
              className="timeline-nav__pill"
              onClick={() => setShowHistoryRounds((value) => !value)}
            >
              {showHistoryRounds ? '折叠历史轮' : `展开历史轮 (${Math.max(rounds.length - 1, 0)})`}
            </button>
            <button
              type="button"
              className="timeline-nav__pill"
              onClick={() => void handleClearMemory()}
            >
              {resetting ? '清零中' : '实验清零'}
            </button>
            <button
              type="button"
              className="timeline-nav__pill"
              onClick={() => void refresh()}
            >
              {refreshing ? '刷新中' : '刷新状态'}
            </button>
          </div>
        </header>

        <section className="workflow-overview">
          <div className="workflow-overview__hero">
            <div className="workflow-overview__question-card workflow-overview__question-card--hero">
              <span className="detail-card__eyebrow">科学问题</span>
              <strong>{displayQuestion || '尚未录入科学问题'}</strong>
            </div>
          </div>

          <div className="timeline-meta__metrics workflow-overview__metrics">
            <div className="metric-pill">
              <span className="metric-pill__label">当前轮次</span>
              <span className="metric-pill__value">{data?.viewModels.currentRoundNumber ?? '--'}</span>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">当前节点</span>
              <span className="metric-pill__value">{processMonitor?.currentStep ?? '--'}</span>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">当前阶段</span>
              <span className="metric-pill__value">{processMonitor?.currentPhase ?? '--'}</span>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">知识材料</span>
              <span className="metric-pill__value">{knowledgeUploadCount || '--'}</span>
            </div>
            <div className="metric-pill">
              <span className="metric-pill__label">数据文件</span>
              <span className="metric-pill__value">{dataCount || '--'}</span>
            </div>
          </div>
        </section>

        {showHypothesisTree && currentRound ? (
          <section className="hypothesis-tree-quick-panel">
            <div className="hypothesis-tree-quick-panel__header">
              <span className="detail-card__eyebrow">假设树速览</span>
              <h2>当前假设树 · 第 {currentRound.roundNumber} 轮</h2>
              <span>当前展示 {currentRound.hypotheses.length} 个竞争假设</span>
            </div>
            {currentRound.hypotheses.length > 0 ? (
              <div className="hypothesis-tree-preview">
                {currentRound.hypotheses.map((item, index) => (
                  <article
                    key={item.id}
                    className={[
                      'hypothesis-badge',
                      `hypothesis-badge--${item.status}`,
                    ].join(' ')}
                  >
                    <span className="hypothesis-badge__id">
                      {item.displayLabel ?? `H${index + 1}`}
                    </span>
                    <span className="hypothesis-badge__label">{item.label}</span>
                    <span className="hypothesis-badge__score">
                      {hypothesisStatusLabel(item.status)} · 支持度 {item.supportScore.toFixed(3)}
                    </span>
                  </article>
                ))}
              </div>
            ) : (
              <p className="timeline-empty-state">当前轮尚未生成假设树。</p>
            )}
          </section>
        ) : null}

        <main className="workflow-board">
          {loading ? (
            <div className="timeline-empty-state">正在读取主流程状态…</div>
          ) : error ? (
            <div className="timeline-empty-state timeline-empty-state--error">{error}</div>
          ) : currentRound ? (
            <>
              <section className="workflow-board__main">
                <div className="workflow-step-rail">
                  {currentRound.nodes.map((node) => (
                    <button
                      key={node.id}
                      type="button"
                      className={`workflow-step workflow-step--${node.status}`}
                      onClick={() => handleOpenNode(currentRound.id, node.id, node.shortLabel)}
                    >
                      <span className="workflow-step__icon">{node.shortLabel}</span>
                      <span className="workflow-step__title">{node.title}</span>
                      <span className="workflow-step__summary">{node.summary}</span>
                    </button>
                  ))}
                </div>

                <div className="workflow-insight-grid workflow-insight-grid--split">
                  <article className="detail-card">
                    <span className="detail-card__eyebrow">当前焦点</span>
                    <h2>{currentRound.stateLabel}</h2>
                    <p>{currentRound.subtitle}</p>
                    {sessionStatus?.status === 'failed' ? (
                      <div className="failure-attribution-card">
                        <strong>失败归因</strong>
                        <p>{sessionStatus.message ?? '当前闭环运行失败，请检查实验协议、字段映射与数据约束。'}</p>
                        <p>
                          失败阶段：{processMonitor?.currentPhase ?? '--'} / {processMonitor?.currentStep ?? '--'}
                        </p>
                      </div>
                    ) : null}
                    <ul className="detail-list">
                      {currentRound.hypotheses.map((item) => (
                        <li key={item.id}>
                          <strong>{item.displayLabel ?? item.id}</strong> {item.label} · 支持度{' '}
                          {item.supportScore.toFixed(3)}
                        </li>
                      ))}
                    </ul>
                    <div className="detail-links">
                      <button
                        type="button"
                        className="detail-link detail-link--button"
                        onClick={() => void handleClearMemory()}
                      >
                        {resetting ? '正在清空假设空间…' : '清空假设空间记忆'}
                      </button>
                    </div>
                  </article>

                  <article className="detail-card">
                    <span className="detail-card__eyebrow">本轮结果</span>
                    <h2>评价摘要</h2>
                    {currentRound.evaluation ? (
                      <div>
                        <ul className="detail-list">
                          <li>实验: {currentRound.evaluation.experimentId ?? '--'}</li>
                          <li>实验类型: {experimentTypeLabel(currentRound.evaluation.experimentType)}</li>
                          <li>结论: {currentRound.evaluation.verdict}</li>
                        </ul>
                        <RunMetricsGrid
                          runs={currentRound.evaluation.runs ?? []}
                          experimentType={currentRound.evaluation.experimentType}
                        />
                        <ContrastRows contrasts={currentRound.evaluation.contrasts ?? []} />
                        <DiagnosticsList diagnostics={currentRound.evaluation.diagnostics} />
                      </div>
                    ) : (
                      <p>当前轮尚未完成执行评价。</p>
                    )}
                  </article>
                </div>
              </section>

              <section className="workflow-history">
                <div className="workflow-history__header">
                  <span className="detail-card__eyebrow">历史轮次</span>
                  <strong>轮次回看</strong>
                </div>
                <div className="workflow-history__list">
                  {displayedRounds.map((round) => (
                    <article key={round.id} className={`workflow-history-card ${round.isCurrent ? 'workflow-history-card--current' : ''}`}>
                      <div className="workflow-history-card__header">
                        <strong>Round {round.roundNumber}</strong>
                        <span>{round.stateLabel}</span>
                      </div>
                      <p className="workflow-history-card__conclusion">
                        {round.conclusion || (round.isCurrent ? '本轮结论待生成' : '本轮尚未归档结论')}
                      </p>
                      <div className="workflow-history-card__meta">
                        <span>假设 {round.hypotheses.length}</span>
                        <span>不确定性 {round.uncertainties.length}</span>
                        <span>实验 {round.experiments.length}</span>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            </>
          ) : (
            <div className="timeline-empty-state">尚未开始实验，请先在中央控制智能体页输入科学问题并上传数据。</div>
          )}
        </main>

        <section className="detail-card detail-card--wide workflow-mission-panel">
          <div className="workflow-mission-panel__header">
            <div>
              <span className="detail-card__eyebrow">Task Definition</span>
              <h2>任务定义</h2>
            </div>
            <span className="detail-card__meta">{mission?.questionType ?? '未定义问题类型'}</span>
          </div>
          <div className="workflow-mission-panel__grid">
            <div className="workflow-mission-panel__cell workflow-mission-panel__cell--wide">
              <strong>科学问题</strong>
              <p>{displayQuestion || '尚未录入科学问题'}</p>
            </div>
            <div className="workflow-mission-panel__group">
              <strong className="workflow-mission-panel__group-title">研究目标与变量</strong>
              <div className="workflow-mission-panel__vars">
                <div>
                  <b>研究目标</b>
                  <span>{mission?.target || '未定义'}</span>
                </div>
                <div>
                  <b>解释变量</b>
                  <span>{mission?.variables.x || '未定义'}</span>
                </div>
                <div>
                  <b>目标变量</b>
                  <span>{mission?.variables.y || mission?.target || '未定义'}</span>
                </div>
                <div>
                  <b>候选特征</b>
                  <span>
                    {(mission?.variables.mCandidates ?? []).length > 0
                      ? mission!.variables.mCandidates.join('、')
                      : '未配置'}
                  </span>
                </div>
              </div>
            </div>
            <div className="workflow-mission-panel__cell">
              <strong>约束条件</strong>
              <ul className="detail-list">
                {(mission?.constraints ?? []).length > 0
                  ? mission!.constraints.map((item, index) => <li key={`constraint-${index}`}>{item}</li>)
                  : <li>未显式配置</li>}
              </ul>
            </div>
            <div className="workflow-mission-panel__cell">
              <strong>评价指标</strong>
              <p>{(mission?.metrics ?? []).length > 0 ? mission!.metrics.join('、') : '未配置'}</p>
            </div>
          </div>
        </section>

        <section className="detail-card detail-card--wide workflow-upload-panel">
          <div className="workflow-upload-panel__header">
            <div>
              <span className="detail-card__eyebrow">Uploaded Files</span>
              <h2>已上传文件</h2>
            </div>
            <span className="detail-card__meta">{dataTableNames.length} 个数据文件</span>
          </div>
          <div className="workflow-upload-panel__grid">
            <div>
              <strong>数据文件</strong>
              {dataTableNames.length > 0 ? (
                <ul className="workflow-mission-panel__files">
                  {dataTableNames.map((name) => <li key={name}>{name}</li>)}
                </ul>
              ) : (
                <p>尚未上传数据文件</p>
              )}
            </div>
            <div>
              <strong>知识材料</strong>
              {(knowledgeBase?.uploads ?? []).length > 0 ? (
                <ul className="detail-list">
                  {knowledgeBase!.uploads.map((item) => <li key={item.id}>{item.fileName}</li>)}
                </ul>
              ) : (
                <p>尚未接入独立知识文件</p>
              )}
            </div>
            <div>
              <strong>推理痕迹</strong>
              <p>{knowledgeCount > 0 ? `当前上下文包含 ${knowledgeCount} 条智能体推理片段` : '暂无推理痕迹'}</p>
            </div>
          </div>
        </section>

        <footer className="timeline-progress">
          <div className="timeline-progress__label">
            第 {data?.viewModels.currentRoundNumber ?? '--'} 轮 · {data?.viewModels.progressPercentage ?? 0}% · 最近刷新{' '}
            {data ? new Date(data.refreshedAt).toLocaleTimeString() : '--'}
          </div>
          <div className="timeline-progress__bar">
            <div className="timeline-progress__fill" style={{ width: `${data?.viewModels.progressPercentage ?? 0}%` }} />
          </div>
        </footer>
      </div>
    </div>
  )
}
