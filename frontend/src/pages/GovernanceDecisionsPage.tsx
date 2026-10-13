import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { submitApprovalAction } from '../api/liveWorkflow'
import { ApprovalOverlay } from '../components/ApprovalOverlay'
import { CandidateExperimentTable } from '../components/CandidateExperimentTable'
import { PageTabs } from '../components/PageTabs'
import { StepConfirmDialog } from '../components/StepConfirmDialog'
import { useTimelineBundle } from '../hooks/useTimelineBundle'
import { filterSelectableCandidateExperiments, getExperimentValidationSnapshot } from '../utils/experimentValidation'
import { toDisplayText } from '../data/realStateLoader'

export function GovernanceDecisionsPage() {
  const navigate = useNavigate()
  const { data, loading, error, refreshing, refresh } = useTimelineBundle()
  const governance = data?.viewModels.governance
  const approvalOverlay = data?.viewModels.approvalOverlay
  const currentRoundNumber =
    Number(data?.viewModels.currentRoundNumber ?? data?.snapshot.process?.current_round ?? 0) || 0
  const candidateRound =
    Number(data?.snapshot.candidateExperiments?.round ?? currentRoundNumber) || currentRoundNumber
  const candidateRelations: any[] = filterSelectableCandidateExperiments(
    (data?.snapshot.candidateExperiments?.candidates ?? []).filter(
      (candidate: any) => Number(candidate.round_id ?? candidateRound) === currentRoundNumber,
    ),
  )
  const roundUncertainties = (data?.snapshot.uncertainties?.records ?? []).filter((item: any) => {
    const fallbackRound =
      Number(data?.snapshot.uncertainties?.current_round ?? currentRoundNumber) || currentRoundNumber
    return Number(item.created_at_round ?? item.current_round ?? fallbackRound) === currentRoundNumber
  })
  const displayText = useMemo(
    () => (text?: string) => toDisplayText(text ?? '', data?.snapshot.plannerInput?.data_dictionary_summary),
    [data?.snapshot.plannerInput?.data_dictionary_summary],
  )
  const [actionBusy, setActionBusy] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [pendingAction, setPendingAction] = useState<{
    action: 'approve' | 'modify' | 'reject' | 'pause'
    payload: { candidateId?: string; humanNotes?: string }
    title: string
    message: string
    sections?: Array<{ label: string; value: string }>
    footer?: string
    confirmLabel: string
  } | null>(null)

  async function submitAction(
    action: 'approve' | 'modify' | 'reject' | 'pause',
    payload: { candidateId?: string; humanNotes?: string },
  ) {
    try {
      setActionBusy(true)
      setActionError(null)
      const result = await submitApprovalAction({
        action,
        candidateId: payload.candidateId ?? approvalOverlay?.recommendation.candidateId,
        humanNotes: payload.humanNotes,
      })
      await refresh()
      if (action === 'approve' || action === 'modify') {
        navigate('/execution')
      } else if (result.stage === 'approval_pending' || result.status === 'awaiting_approval') {
        navigate('/approval')
      }
    } catch (error) {
      setActionError(error instanceof Error ? error.message : '审批动作提交失败。')
    } finally {
      setActionBusy(false)
    }
  }

  async function queueApprovalAction(
    action: 'approve' | 'modify' | 'reject' | 'pause',
    payload: { candidateId?: string; humanNotes?: string },
  ) {
    const candidateId = payload.candidateId ?? approvalOverlay?.recommendation.candidateId ?? '当前推荐实验'
    const candidatePreview = candidateRelations.find((item: any) => item.experiment_id === candidateId)
    const validation = getExperimentValidationSnapshot(candidatePreview)
    const treatmentVariables = validation.treatmentVariables.join('、') || '待系统生成'
    const controlVariables = validation.controlVariables.join('、') || '待系统生成'
    const isBaselineExperiment = validation.experimentMode === 'baseline'
    const hasFeatureDifference = validation.hasFeatureDifference
    const scientificQuestion = candidatePreview?.scientific_question ?? candidatePreview?.purpose ?? '当前实验计划将围绕关键不确定性推进。'
    const contentMap = {
      approve: {
        title: '是否批准当前实验并进入执行？',
        message: `确认执行 ${candidateId}？`,
        sections: [
          { label: '实验', value: candidateId },
          { label: '实验目的', value: scientificQuestion },
          ...(isBaselineExperiment
            ? [{ label: '对照组变量', value: treatmentVariables }]
            : [
                { label: '对照组变量', value: controlVariables },
                { label: '实验组变量', value: treatmentVariables },
                { label: '变量差异校验', value: hasFeatureDifference ? '通过' : '未通过' },
              ]),
        ],
        footer: '确认后系统将生成执行协议并自动进入实验执行页面。',
        confirmLabel: '确认批准',
      },
      modify: {
        title: '是否按修改意见继续审批？',
        message: `实验 ${candidateId} 将带着你的修改意见继续推进。`,
        sections: [
          { label: '实验', value: candidateId },
          { label: '实验目的', value: scientificQuestion },
          { label: '实验组变量', value: treatmentVariables },
        ],
        footer: '确认后系统将重新写入协议并自动进入实验执行页面。',
        confirmLabel: '确认修改',
      },
      reject: {
        title: '是否拒绝当前实验并返回重选？',
        message: `系统将拒绝 ${candidateId}，并回到候选实验重选流程。`,
        confirmLabel: '确认拒绝',
      },
      pause: {
        title: '是否暂停当前审批流程？',
        message: '系统将暂停当前候选实验审批，等待后续人工继续处理。',
        confirmLabel: '确认暂停',
      },
    }

    setPendingAction({
      action,
      payload,
      ...contentMap[action],
    })
  }

  return (
    <div className="timeline-shell">
      <div className="timeline-shell__frame timeline-shell__frame--detail">
        <div className="timeline-topbar">
          <PageTabs />
        </div>

        <section className="workspace-page">
          {loading ? (
            <div className="timeline-empty-state">正在读取治理与决策状态…</div>
          ) : error || !governance ? (
            <div className="timeline-empty-state timeline-empty-state--error">
              {error ?? '治理与决策数据不可用。'}
            </div>
          ) : (
            <>
              {approvalOverlay ? (
                <>
                  <section className="detail-card detail-card--wide">
                    <span className="detail-card__eyebrow">Inline Approval</span>
                    <h2>页内审批操作</h2>
                    <ApprovalOverlay
                      data={approvalOverlay}
                      open
                      inline
                      onClose={() => undefined}
                      onAction={queueApprovalAction}
                      busy={actionBusy}
                    />
                  </section>
                  {actionError ? <div className="timeline-empty-state timeline-empty-state--error">{actionError}</div> : null}
                </>
              ) : null}

              <header className="workspace-page__header">
                <div>
                  <span className="timeline-header__eyebrow">Approval</span>
                  <h1>批准实验决策</h1>
                  <p>突出当前最优候选实验的综合价值，并由人类 PI 做出是否采纳的判断。</p>
                </div>

                <div className="detail-page__meta">
                  <button
                    type="button"
                    className="detail-link detail-link--button"
                    onClick={() => void refresh()}
                  >
                    {refreshing ? '刷新中…' : '刷新'}
                  </button>
                  <div className="status-chip">
                    <span>待审批实验</span>
                    <strong>{governance.pendingApproval?.candidateId ?? '--'}</strong>
                  </div>
                  <div className="status-chip">
                    <span>当前阶段</span>
                    <strong>{governance.processSummary.currentPhase ?? '--'}</strong>
                  </div>
                  <div className="status-chip">
                    <span>当前步骤</span>
                    <strong>{governance.processSummary.currentStep ?? '--'}</strong>
                  </div>
                </div>
              </header>

              <section className="detail-card detail-card--wide">
                <span className="detail-card__eyebrow">Uncertainty Mapping</span>
                <h2>不确定性到候选实验的推导关系</h2>
                <CandidateExperimentTable
                  candidates={candidateRelations}
                  treeNodes={data?.snapshot.hypothesisTree?.nodes ?? []}
                  uncertaintyRecords={roundUncertainties}
                  displayText={displayText}
                />
              </section>

              <section className="detail-card detail-card--wide">
                <span className="detail-card__eyebrow">Candidate Matrix</span>
                <h2>当前候选实验综合价值</h2>
                <div className="approval-candidate-strip">
                  {approvalOverlay?.candidates.map((candidate) => (
                    <article
                      key={candidate.experimentId}
                      className={`approval-candidate-card ${candidate.recommended ? 'approval-candidate-card--recommended' : ''}`}
                    >
                      <div className="approval-candidate-card__header">
                        <strong>{candidate.experimentId}</strong>
                        {candidate.recommended ? <span>最优</span> : null}
                      </div>
                      <div className="approval-candidate-card__utility">{candidate.utility.toFixed(3)}</div>
                      <div className="approval-candidate-card__metrics">
                        <span>预期区分度 {candidate.informationGain.toFixed(3)}</span>
                        <span>预期性能 {candidate.performanceGain.toFixed(3)}</span>
                        <span>Risk {candidate.risk.toFixed(3)}</span>
                        <span>Cost {candidate.cost.toFixed(3)}</span>
                      </div>
                    </article>
                  ))}
                </div>
              </section>

            </>
          )}
        </section>
        <StepConfirmDialog
          open={Boolean(pendingAction)}
          title={pendingAction?.title ?? '确认审批动作'}
          message={pendingAction?.message ?? '请确认当前审批动作。'}
          sections={pendingAction?.sections}
          footer={pendingAction?.footer}
          confirmLabel={pendingAction?.confirmLabel ?? '确认'}
          onCancel={() => setPendingAction(null)}
          onConfirm={() => {
            if (!pendingAction) {
              return
            }
            const nextAction = pendingAction
            setPendingAction(null)
            void submitAction(nextAction.action, nextAction.payload)
          }}
        />
      </div>
    </div>
  )
}
