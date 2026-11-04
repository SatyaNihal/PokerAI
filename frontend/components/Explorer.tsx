'use client';
import {useEffect,useState} from 'react';
import {api,cardName,historyName,type Agent,type GameName,type Info,type Policy} from '../lib/api';
import {Card,ErrorBox,Loading,Probabilities} from './common';
import HoldemExplorerControls from './HoldemExplorerControls';

export default function Explorer({agents}:{agents:Agent[]}) {
  const [game,setGame]=useState<GameName>('holdem');
  const [catalog,setCatalog]=useState<Info[]>([]);
  const [selected,setSelected]=useState<Info|null>(null);
  const [policies,setPolicies]=useState<Policy[]>([]);
  const [error,setError]=useState('');
  const [loading,setLoading]=useState(true);
  const [retry,setRetry]=useState(0);
  useEffect(()=>{let cancelled=false;setLoading(true);setError('');setPolicies([]);api<Info[]>(`/information-sets?game=${game}`).then(data=>{if(!cancelled){setCatalog(data);setSelected(data[0]);}}).catch(e=>{if(!cancelled){setError(e.message);setLoading(false);}});return()=>{cancelled=true;};},[game,retry]);
  useEffect(()=>{if(!selected)return;let cancelled=false;setLoading(true);setError('');const choices=agents.filter(a=>a.game===game);Promise.all(choices.map(a=>api<Policy>(`/agents/${a.id}/policy?information_set=${encodeURIComponent(selected.key)}`))).then(data=>{if(!cancelled){setPolicies(data);setLoading(false);}}).catch(e=>{if(!cancelled){setError(e.message);setLoading(false);}});return()=>{cancelled=true;};},[selected,agents,game]);

  function selectField(field:'private_card'|'public_card', value:number|null) {
    if(!selected)return;
    const match=catalog.find(v=>v[field]===value && v.player===selected.player && v.private_card===(field==='private_card'?value:selected.private_card) && v.public_card===(field==='public_card'?value:selected.public_card) && JSON.stringify(v.history)===JSON.stringify(selected.history)) || catalog.find(v=>v[field]===value && (field==='private_card'||v.private_card===selected.private_card)) || catalog.find(v=>v[field]===value);
    if(match)setSelected(match);
  }
  const histories=selected ? catalog.filter(v=>v.private_card===selected.private_card && v.public_card===selected.public_card) : [];
  return <>
    <section className="panel explorer-controls">
      <label>Game<select aria-label="Explorer game" value={game} onChange={e=>{setSelected(null);setGame(e.target.value as GameName);}}><option value="holdem">Texas Hold’em · limit</option><option value="leduc">Leduc Hold’em</option><option value="kuhn">Kuhn Poker</option></select></label>
      {selected && game==='holdem' && <HoldemExplorerControls selected={selected} catalog={catalog} onSelect={setSelected}/>}
      {selected && game!=='holdem' && <><label>Private card<select aria-label="Private card" value={selected.private_card} onChange={e=>selectField('private_card',Number(e.target.value))}>{Array.from(new Set(catalog.map(v=>v.private_card))).sort((a,b)=>a-b).map(c=><option key={c} value={c}>{cardName(game,c)}</option>)}</select></label>
      {game==='leduc' && <label>Public card<select aria-label="Public card" value={selected.public_card ?? -1} onChange={e=>selectField('public_card',Number(e.target.value)===-1?null:Number(e.target.value))}>{[null,...[0,1,2,3,4,5].filter(c=>c!==selected.private_card)].map(c=><option key={c??-1} value={c??-1}>{cardName(game,c)}</option>)}</select></label>}
      <label className="history-select">Betting history<select aria-label="Betting history" value={selected.key} onChange={e=>setSelected(catalog.find(v=>v.key===e.target.value)!)}>{histories.map(v=><option key={v.key} value={v.key}>P{v.player} · {historyName(v)}</option>)}</select></label></>}
    </section>
    {error && <ErrorBox error={error} retry={()=>setRetry(v=>v+1)}/>}
    {selected && <div className="observation-strip"><div className="observation-cards">{(selected.private_cards||[selected.private_card]).map(c=><Card key={c} game={game} card={c} small/>)}{game==='holdem' && selected.public_cards?.map(c=><Card key={c} game={game} card={c} small/>)}{game==='leduc' && <Card game={game} card={selected.public_card} small/>}</div><div><span className="eyebrow">Observed state</span><p>Player {selected.player} · {historyName(selected)}</p><code className="info-key">{selected.key}</code></div></div>}
    {loading ? <Loading label="Querying saved policies…"/> : <div className="policy-grid">{policies.map(policy=>{const agent=agents.find(a=>a.id===policy.agent_id)!;return <section className={`panel policy-card ${agent.algorithm}`} key={policy.agent_id}><div className="section-heading"><h2>{agent.name}</h2><span className={`method-dot ${agent.algorithm}`}/></div><p className="muted">{agent.algorithm==='deep_cfr'?'Neural average policy':'Reach-weighted average policy'}</p><Probabilities info={policy.information_state} probabilities={policy.probabilities}/><a className="record-link" href={`/experiments/${policy.experiment_id}`} target="_blank" rel="noreferrer">Training record</a></section>;})}</div>}
    {policies[0] && <details className="panel encoder-details"><summary>Inspect the model input</summary><p>One-hot player and physical cards, followed by the public betting history. No hand-strength features.</p><code>{JSON.stringify(policies[0].tensor)}</code><small>Encoder: {policies[0].encoder_version}</small></details>}
  </>;
}
