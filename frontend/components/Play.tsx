'use client';
import {useEffect,useState} from 'react';
import {api,cardName,historyName,number,type Agent,type Analysis,type BoardRunout,type Game,type GameName} from '../lib/api';
import {Card,ErrorBox,Loading,Probabilities} from './common';

export default function Play({agents,onExplore}:{agents:Agent[];onExplore:()=>void}) {
  const [game,setGame]=useState<GameName>('holdem');
  const [agentId,setAgentId]=useState('holdem-deep_cfr');
  const [seat,setSeat]=useState('random');
  const [hand,setHand]=useState<Game|null>(null);
  const [analysis,setAnalysis]=useState<Analysis|null>(null);
  const [runout,setRunout]=useState<BoardRunout|null>(null);
  const [showRunout,setShowRunout]=useState(false);
  const [busy,setBusy]=useState(false);
  const [error,setError]=useState('');
  const choices=agents.filter(a=>a.game===game);
  const selected=agents.find(a=>a.id===agentId&&a.game===game);
  const active=hand?.status==='playing';
  const holdem=game==='holdem';
  useEffect(()=>{if(choices.length&&!choices.some(a=>a.id===agentId))setAgentId(choices[0].id);},[agents,game,agentId]);

  async function start(){setBusy(true);setError('');setAnalysis(null);setRunout(null);setShowRunout(false);try{setHand(await api<Game>('/games',{game,agent_id:agentId,human_seat:seat==='random'?null:Number(seat)}));}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  async function act(action:number){if(!hand)return;setBusy(true);setError('');try{setHand(await api<Game>(`/games/${hand.id}/actions`,{action,version:hand.version}));}catch(e){setError((e as Error).message);try{setHand(await api<Game>(`/games/${hand.id}`));}catch{/* Keep the original error visible. */}}finally{setBusy(false);}}
  async function inspect(){if(!hand)return;setBusy(true);setError('');try{setAnalysis(await api<Analysis>(`/games/${hand.id}/analysis`));}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  async function reveal(){
    if(!hand||hand.status!=='complete')return;
    if(showRunout){setShowRunout(false);return;}
    if(runout?.game_id===hand.id){setShowRunout(true);return;}
    setBusy(true);setError('');
    try{setRunout(await api<BoardRunout>(`/games/${hand.id}/runout`));setShowRunout(true);}
    catch(e){setError((e as Error).message);}
    finally{setBusy(false);}
  }
  const payoff=hand?.returns?.[hand.human_seat];
  const aiCards=analysis?.ai_cards||[analysis?.ai_card];
  const revealed=showRunout&&runout!==null&&runout.game_id===hand?.id;
  const board=revealed?runout.public_cards:holdem?(hand?.public_cards||[]):hand?.public_card!=null?[hand.public_card]:[];
  const canReveal=hand?.status==='complete'&&game!=='kuhn'&&(holdem?(hand.public_cards?.length||0)<5:hand.public_card==null);
  const title=holdem?'Texas Hold’em · fixed limit':game==='leduc'?'Leduc Hold’em':'Kuhn Poker';
  return <>
    <div className="setup-bar panel">
      <label>Game<select aria-label="Game" value={game} disabled={active||busy} onChange={e=>{setGame(e.target.value as GameName);setHand(null);setAnalysis(null);}}><option value="holdem">Texas Hold’em · limit</option><option value="leduc">Leduc Hold’em</option><option value="kuhn">Kuhn Poker</option></select></label>
      <label>Opponent<select aria-label="Opponent" value={agentId} disabled={active||busy} onChange={e=>{setAgentId(e.target.value);setHand(null);setAnalysis(null);}}>{choices.map(a=><option key={a.id} value={a.id}>{a.name}</option>)}</select></label>
      <label>Your position<select aria-label="Your position" value={seat} disabled={active||busy} onChange={e=>setSeat(e.target.value)}><option value="random">Either seat</option><option value="0">{holdem?'Button · small blind (P0)':'First to act · P0'}</option><option value="1">{holdem?'Big blind (P1)':'Second to act · P1'}</option></select></label>
      <button className="primary" onClick={start} disabled={busy||!selected||selected.game!==game}>{hand?(active?'Restart hand':'Play again'):'Deal a hand'}</button>
    </div>
    {holdem&&<p className="research-note">Limit Hold’em uses approximate evaluation. Exact exploitability is unavailable.</p>}
    {error&&<ErrorBox error={error}/>}
    <div className="play-grid">
      <div><div className="table-shell">
        <div className="table-topline"><span className="eyebrow">{title}</span><span className="live-label">{hand?(holdem?['Preflop','Flop','Turn','River'][hand.round-1]:`Round ${hand.round}`):'Not started'}</span></div>
        <div className="poker-table" aria-label="Poker table">
          {!hand?<div className="empty-table"><h2>No hand in progress</h2><p>Choose a game and opponent, then deal.</p></div>:<>
            <div className="opponent-seat"><div className="seat-tag">{selected?.name} <span>· P{1-hand.human_seat}</span></div><div className="hole-cards">{Array.from({length:holdem?2:1},(_,i)=><Card key={i} game={game} card={aiCards[i]} hidden={!analysis}/>)}</div><div className="table-caption">{hand.stacks?`${number(hand.stacks[1-hand.human_seat])} chips`:`${hand.contributions[1-hand.human_seat]} committed`}</div></div>
            <div className={`table-middle ${holdem?'holdem-middle':''}`}><div className="pot">Pot <strong>{hand.pot}</strong></div>{holdem?<div className="community holdem-board" data-testid="community" aria-label="Community cards">{Array.from({length:5},(_,i)=><Card key={i} game={game} card={board[i]} small/>)}</div>:game==='leduc'?<div className="community" data-testid="community"><Card game={game} card={board[0]} small/><span>Community card</span></div>:<div className="kuhn-note">One private card.<br/>One betting round.</div>}</div>
            {revealed&&<p className="runout-note" role="status">Remaining cards revealed after the hand. Result unchanged.</p>}
            <div className="human-seat"><div className="hole-cards" aria-label="Your hole cards">{(hand.private_cards||[hand.private_card]).map(c=><Card key={c} game={game} card={c}/>)}</div><div className="seat-tag">You <span>· P{hand.human_seat}</span></div><div className="table-caption">{hand.stacks?`${number(hand.stacks[hand.human_seat])} chips`:`${hand.contributions[hand.human_seat]} committed`}</div></div>
          </>}
        </div>
        <div className="action-area" aria-live="polite">
          {busy?<Loading label={hand?'Updating the hand…':'Dealing…'}/>:!hand?<span className="muted">Ready to deal.</span>:hand.status==='complete'?<div className="hand-result" data-testid="hand-result"><div><span className="eyebrow">HAND COMPLETE</span><strong>{payoff===0?'Split pot.':payoff!>0?'You won the hand.':'You lost the hand.'}</strong><span className={payoff!>=0?'positive':'negative'}>{payoff!>0?'+':''}{number(payoff!)} chips</span></div><div className="posthand-actions">{canReveal&&<button className="secondary" onClick={reveal} aria-pressed={Boolean(revealed)}>{revealed?'Hide remaining cards':'Show remaining cards'}</button>}<button className="secondary" onClick={inspect}>View AI decisions</button></div></div>:<><span className="turn-label">Your move <small>Choose a legal action</small></span><div className="action-buttons">{hand.legal_actions.map(a=><button key={a.id} data-action={a.id} className={a.label==='Fold'?'fold-button':'action-button'} onClick={()=>act(a.id)}>{a.label}{a.cost>0&&<small>{a.cost} chip{a.cost===1?'':'s'}</small>}</button>)}</div></>}
        </div>
      </div>
      {analysis&&<section className="panel analysis-panel" aria-label="Post-game analysis"><div className="section-heading"><h2>AI decisions</h2><span className="pill">AI cards · {(analysis.ai_cards||[analysis.ai_card]).map(c=>cardName(game,c)).join(' ')}</span></div><p className="muted">These are the probabilities recorded at decisions the AI actually faced.</p>{analysis.decisions.length===0?<p>The AI did not face a decision in this hand.</p>:analysis.decisions.map((d,i)=><div className="decision-row" key={i}><div><span className="eyebrow">Decision {i+1}</span><p>Private: <strong>{(d.information_state.private_cards||[d.information_state.private_card]).map(c=>cardName(game,c)).join(' ')}</strong>{game!=='kuhn'&&<> · Public: <strong>{holdem?(d.information_state.public_cards?.map(c=>cardName(game,c)).join(' ')||'Undealt'):cardName(game,d.information_state.public_card)}</strong></>}</p><code>{historyName(d.information_state)}</code></div><Probabilities info={d.information_state} probabilities={d.probabilities} chosen={d.action}/></div>)}</section>}
      </div>
      <aside className="play-aside">
        <section className="panel history-panel"><div className="section-heading"><h2>Action history</h2></div>{!hand?.history.length?<p className="muted">No actions yet.</p>:<ol className="history-list">{hand.history.map((a,i)=><li key={i}><span className={`history-dot ${a.player===hand.human_seat?'you-dot':''}`}/><span>{a.player===hand.human_seat?'You':selected?.name}<small>{holdem?['Preflop','Flop','Turn','River'][a.round-1]:`Round ${a.round}`}</small></span><strong>{a.label}</strong></li>)}</ol>}</section>
        {selected&&<section className="panel model-panel"><h2>{selected.name}</h2>{holdem?<><div className="metric-mini"><span>Held-out payoff vs. uniform</span><strong>{number(selected.metrics.uniform_payoff,4)}</strong><small>chips per hand · registered demo seed</small></div><p>This baseline score does not measure distance from optimal play.</p></>:<><div className="metric-mini"><span>Exact exploitability</span><strong>{number(selected.metrics.exploitability,6)}</strong><small>chips per hand · average policy</small></div><p>Measured against an exact best response across both seats.</p></>}<button className="text-button" onClick={onExplore}>Explore policy</button><a className="record-link" href={`/experiments/${selected.experiment_id}`} target="_blank" rel="noreferrer">Experiment record</a><code className="run-id" title={selected.experiment_id}>{selected.experiment_id}</code></section>}
        <p className="aside-note">AI cards and decision probabilities are revealed after the hand.</p>
      </aside>
    </div>
  </>;
}
