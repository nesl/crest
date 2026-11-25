---
hide:
  - navigation
  - toc
---

<div class="crest-home">
  <section class="crest-hero" aria-labelledby="crest-home-title">
    <div class="crest-hero-copy">
      <h1 id="crest-home-title">Inside a CREST trial</h1>
      <p>Follow a candidate from the optimizer to the target board and back. CREST brings task quality, memory, latency, and energy into the decision to prune a trial or continue the search.</p>
      <div class="crest-actions">
        <a class="crest-button" href="run/first-study/">Run your first study <span aria-hidden="true">→</span></a>
        <a class="crest-text-link" href="concepts/">How CREST works <span aria-hidden="true">↗</span></a>
      </div>
    </div>
    <figure class="trial-figure">
      <div class="trial-map">
<img src="images/crest-trial-flow.png" alt="The optimizer samples a candidate. The model family builds it, and the HIL server sends it to the target backend for compilation and measurement. Proxy, quality, latency, energy, RAM, flash, and status feed back into the trial decision.">
<a class="trial-node" href="config/search/" title="Optimizer client — search objectives" aria-label="Optimizer client — search objectives" style="left:6.6667%;top:0.0000%;width:24.8718%;height:18.3150%"></a>
<a class="trial-node" href="api/models/" title="Model family — building candidates" aria-label="Model family — building candidates" style="left:41.0256%;top:0.0000%;width:24.8718%;height:18.3150%"></a>
<a class="trial-node" href="run/hardware/" title="HIL server — hardware setup" aria-label="HIL server — hardware setup" style="left:75.0000%;top:0.0000%;width:25.0000%;height:18.3150%"></a>
<a class="trial-node" href="config/" title="Proxies — desktop study configurations" aria-label="Proxies — desktop study configurations" style="left:47.0513%;top:23.6264%;width:12.8205%;height:15.2015%"></a>
<a class="trial-node" href="api/tasks/" title="Task — training and evaluation" aria-label="Task — training and evaluation" style="left:6.6667%;top:46.7033%;width:24.8718%;height:17.9487%"></a>
<a class="trial-node" href="api/devices/" title="Target backend — program and run a trial" aria-label="Target backend — program and run a trial" style="left:40.8974%;top:46.5201%;width:25.0000%;height:18.1319%"></a>
<a class="trial-node" href="extend/boards/" title="Target backend — compile firmware" aria-label="Target backend — compile firmware" style="left:75.0000%;top:46.5201%;width:25.0000%;height:18.1319%"></a>
<a class="trial-node" href="run/results/" title="Quality metrics — read study results" aria-label="Quality metrics — read study results" style="left:12.8205%;top:70.6960%;width:12.8205%;height:15.2015%"></a>
<a class="trial-node" href="concepts/measurement/" title="Latency and energy — measurement windows" aria-label="Latency and energy — measurement windows" style="left:47.1795%;top:70.6960%;width:12.8205%;height:15.2015%"></a>
<a class="trial-node" href="config/target/" title="RAM and flash — compiler-reported metrics" aria-label="RAM and flash — compiler-reported metrics" style="left:81.1538%;top:70.6960%;width:12.8205%;height:15.2015%"></a>
      </div>
      <figcaption>Select a box to explore the docs. <a href="images/crest-trial-flow.png">View full-size diagram</a></figcaption>
    </figure>
  </section>

  <section class="crest-pathways" aria-labelledby="crest-pathways-title">
    <h2 id="crest-pathways-title">Explore the documentation</h2>
    <div class="crest-cards">
      <a href="run/" class="crest-card">
        <h3>Run studies <span aria-hidden="true">↗</span></h3>
        <p>Set up a study, connect a target board, and inspect or replay the results.</p>
      </a>
      <a href="extend/" class="crest-card">
        <h3>Extend CREST <span aria-hidden="true">↗</span></h3>
        <p>Add a model family or adapt the framework to another workload or target.</p>
      </a>
      <a href="api/" class="crest-card">
        <h3>API reference <span aria-hidden="true">↗</span></h3>
        <p>Look up the Python interfaces for datasets, tasks, models, and boards.</p>
      </a>
    </div>
  </section>

</div>

## Built-in example workloads and targets

<!-- include: README.md#Supported workloads and hardware -->
