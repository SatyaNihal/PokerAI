import {cardName,historyName,type Info} from '../lib/api';

export default function HoldemExplorerControls({selected,catalog,onSelect}:{selected:Info;catalog:Info[];onSelect:(info:Info)=>void}) {
  const own=selected.private_cards||[];
  const board=selected.public_cards||[];
  function change(which:'private'|'public',slot:number,value:number){
    const privateCards=[...own];let publicCards=[...board];
    (which==='private'?privateCards:publicCards)[slot]=value;
    privateCards.sort((a,b)=>a-b);
    publicCards=[...publicCards.slice(0,3).sort((a,b)=>a-b),...publicCards.slice(3)];
    onSelect({...selected,private_card:privateCards[0],public_card:publicCards[0]??null,private_cards:privateCards,public_cards:publicCards,
      key:'holdem:'+JSON.stringify([selected.player,privateCards,publicCards,selected.history])});
  }
  function selector(which:'private'|'public',slot:number,value:number,label:string){
    const used=[...own,...board].filter(c=>c!==value);
    return <label key={`${which}-${slot}`}>{label}<select aria-label={label} value={value} onChange={e=>change(which,slot,Number(e.target.value))}>{Array.from({length:52},(_,c)=>c).filter(c=>!used.includes(c)).map(c=><option key={c} value={c}>{cardName('holdem',c)}</option>)}</select></label>;
  }
  return <div className="holdem-explorer-inputs"><label>Observed decision<select aria-label="Holdem observed decision" value={catalog.some(v=>v.key===selected.key)?selected.key:''} onChange={e=>onSelect(catalog.find(v=>v.key===e.target.value)!)}><option value="" disabled>Edited observable cards</option>{catalog.map(v=><option key={v.key} value={v.key}>P{v.player} · {v.private_cards?.map(c=>cardName('holdem',c)).join(' ')} · {historyName(v)}</option>)}</select></label><div className="holdem-card-inputs">{own.map((c,i)=>selector('private',i,c,`Hole card ${i+1}`))}{board.map((c,i)=>selector('public',i,c,`Board card ${i+1}`))}</div><p className="chart-note">Start from a recorded valid betting history, then change its observable cards. The saved network evaluates edited states directly; this catalog samples training decisions and is not the entire game.</p></div>;
}
