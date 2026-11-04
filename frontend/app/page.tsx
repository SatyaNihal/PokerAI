'use client';
import {useEffect,useState} from 'react';
import {api,type Agent} from '../lib/api';
import {ErrorBox,Loading} from '../components/common';
import Play from '../components/Play';
import Explorer from '../components/Explorer';
import Dashboard from '../components/Dashboard';
import HoldemDashboard from '../components/HoldemDashboard';

type Tab='play'|'explorer'|'dashboard';
const tabs: [Tab, string][] = [
  ['play', 'Play'],
  ['explorer', 'Decision Explorer'],
  ['dashboard', 'Research Dashboard'],
];
const headings: Record<Tab, {title: string; description: string}> = {
  play: {title: 'Play poker', description: 'Play a hand against a self-play agent.'},
  explorer: {title: 'Decision explorer', description: 'Compare policies for the same cards and betting history.'},
  dashboard: {title: 'Training results', description: 'Recorded experiments, policy evaluations, and head-to-head results.'},
};
export default function Page() {
  const [tab,setTab]=useState<Tab>('play');
  const [visited,setVisited]=useState<Tab[]>(['play']);
  const [agents,setAgents]=useState<Agent[]>([]);
  const [error,setError]=useState('');
  const [retry,setRetry]=useState(0);
  useEffect(()=>{api<Agent[]>('/agents').then(setAgents).catch(e=>setError(e.message));},[retry]);
  useEffect(()=>{const hash=window.location.hash.slice(1);if(['play','explorer','dashboard'].includes(hash)){setTab(hash as Tab);setVisited(v=>Array.from(new Set([...v,hash as Tab])));}},[]);
  function navigate(value:Tab){setTab(value);setVisited(v=>Array.from(new Set([...v,value])));window.history.replaceState(null,'',`#${value}`);}
  return <div className="app-shell">
    <header className="site-header">
      <a className="wordmark" href="#play" onClick={e=>{e.preventDefault();navigate('play');}}>Poker AI</a>
      <nav aria-label="Main navigation">
        {tabs.map(([value,label])=><button key={value} aria-current={tab===value?'page':undefined} onClick={()=>navigate(value)}>{label}</button>)}
      </nav>
    </header>
    <main>
      <div className="page-heading">
        <h1>{headings[tab].title}</h1>
        <p>{headings[tab].description}</p>
      </div>
      {error ? <ErrorBox error={error} retry={()=>{setError('');setRetry(v=>v+1);}}/> : !agents.length ? <Loading label="Loading trained agents…"/> : <><section hidden={tab!=='play'}><Play agents={agents} onExplore={()=>navigate('explorer')}/></section>{visited.includes('explorer') && <section hidden={tab!=='explorer'}><Explorer agents={agents}/></section>}{visited.includes('dashboard') && <section hidden={tab!=='dashboard'}><Dashboard/><HoldemDashboard/></section>}</>}
    </main>
    <footer className="site-footer"><span>Kuhn · Leduc · Limit Texas Hold’em</span><a href="/docs" target="_blank" rel="noreferrer">API reference</a></footer>
  </div>;
}
