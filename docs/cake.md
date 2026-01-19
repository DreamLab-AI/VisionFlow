# The AI Cake Analogy: From Slices to Supercake

*An exploration of the AI services landscape through confectionery metaphors*

---

## The Foundation Model Layer: The Cake Itself

Every AI service begins with a foundation model—the cake itself. Each has its distinct flavor, texture, and character:

### The Bakery Lineup

| Foundation Model | Cake Type | Character |
|-----------------|-----------|-----------|
| **Claude (Anthropic)** | Fruitcake | Dense, nutty, and lasts forever. Constitutional ingredients prevent it from going rogue at your dinner party. *"Let me think about that"* translates to 12 delicious seconds of internal monologue before serving. The thinking emoji made edible. |
| **GPT-4 (OpenAI)** | Victoria Sponge | The classic everyone secretly serves at their events. Reliable, crowd-pleasing, and occasionally hiccups *"I'm sorry, I can't help with that recipe"* when asked to make anything controversial. Still the default cake for people who don't know what cake is. |
| **Gemini (Google)** | Mille-Feuille | A thousand layers of flaky goodness—try to pick one up and it might just tell you it can't show you the image. Sometimes the layers don't quite stick together, and Google might discontinue this bakery any day now, but boy is it photogenic. Comes in Ultra, Pro, and Flash sizes. The Flash version is actually quite good, which surprised everyone including Google. |
| **Grok (xAI)** | Carrot Cake | Unexpectedly spicy with a cream cheese frosting of *"actually, let me tell you what really happened."* The only cake that will argue with you about politics and then help you plan a space mission. Now with 100% more posting history. |
| **DeepSeek** | Baumkuchen (Tree Cake) | Laboriously layered by hand in a factory that definitely didn't cost $6M to build. Each ring shows its work, proving that sometimes you can bake a masterpiece with just a whisk, a dream, and possibly state subsidies. The thinking is visible and the reasoning is *chef's kiss*. |
| **Perplexity** | Sachertorte | Rich Austrian chocolate with apricot citation jam. *"Here's your slice, and here are the seven bakeries I verified this recipe from."* The only cake that refuses to make anything up. Academic integrity in pastry form. |
| **Llama (Meta)** | Pound Cake | Open recipe that anyone can download and somehow make even better. Meta keeps releasing new versions while you're still trying to figure out what to do with the 70-pound cake you just baked on your gaming PC. Now available in 3.2 flavors. |
| **Mistral** | Croissant-Cake | French fusion that's lighter than expected and somehow wins every bake-off despite being a fraction of the size. *"Oui, we are small, but we are merci beaucoup efficient."* Proof that European engineering can do more with less. |
| **Phi-3 (Microsoft)** | Mini Cupcake | Does 90% of what the full-size cake does, but fits in your laptop and barely crumbs on your keyboard. Microsoft's adorable attempt at efficiency. Surprisingly capable for something you can run on a toaster. |
| **Cohere Command** | Wedding Cake | Built for enterprise, sturdy enough to withstand corporate mergers. Won't hallucinate a prenup but will definitely summarize the terms and conditions. Enterprise pricing, enterprise portions. |
| **Ollama/Local LLMs** | Homemade Sourdough Cake | You spent 47 hours fermenting the starter and it still tastes like it was baked on a potato. Artisanal, experimental, and definitely not running on the GPU you promised your spouse you'd use for gaming. But it's *yours*, and nobody can take that away. |

---

## The Icing Layer: User & Corporate Content

The icing transforms a generic cake into something personal. This is the user's data, corporate knowledge, fine-tuning, and context.

### Icing Thickness Matters

| Icing Level | Description | Example |
|-------------|-------------|---------|
| **Bare** | Raw model, no customization | Generic ChatGPT with no context |
| **Dusting** | Light system prompt | "You are a helpful assistant" |
| **Thin Glaze** | Basic RAG, some documents | Company FAQ chatbot |
| **Medium Buttercream** | Rich context, fine-tuning | Domain-specific assistant with training |
| **Thick Fondant** | Full knowledge base, ontology-backed | Enterprise knowledge graph with semantic rules |

**The Truth About Icing:**
- Too thin: The cake is generic, forgettable
- Too thick: Overwhelms the flavor, sluggish responses
- Just right: Complements the cake, creates unique identity

*Most AI services give you a dusting and call it done.*

---

## The Cherry: The Interface

The cherry on top is how users interact with the cake. It's the final flourish that makes the experience complete.

### Cherry Varieties

| Interface Type | Cherry Style | Examples |
|---------------|--------------|----------|
| **Chat box** | Maraschino (preserved) | ChatGPT, Claude.ai, Gemini |
| **API only** | No cherry (cake by mail) | OpenAI API, Anthropic API |
| **Voice assistant** | Candied cherry | Siri, Alexa with AI |
| **Embedded copilot** | Cherry in the batter | GitHub Copilot, Cursor |
| **3D immersive** | Fresh cherry, hand-picked | VisionFlow, Spatial Computing |
| **Multi-modal studio** | Cherry compote with vanilla | IRIS (Intelligent Realtime Integrated Studio) |

---

## The Problem: Everyone's Selling Slices

The current AI services landscape is a patisserie of disappointment:

```
┌─────────────────────────────────────────────────────┐
│                CURRENT AI SERVICES                   │
│                                                      │
│   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐      │
│   │ 🍰  │  │ 🍰  │  │ 🍰  │  │ 🍰  │  │ 🍰  │      │
│   │slice│  │slice│  │slice│  │slice│  │slice│      │
│   └─────┘  └─────┘  └─────┘  └─────┘  └─────┘      │
│                                                      │
│   ChatGPT   Claude   Gemini   Grok    Perplexity   │
│                                                      │
│   Single    Single   Single   Single   Single       │
│   Model     Model    Model    Model    Model        │
│   ────────────────────────────────────────────      │
│   Thin      Thin     Thin     Quirky   Citation     │
│   Icing     Icing    Icing    Icing    Icing        │
│   ────────────────────────────────────────────      │
│   Chat      Chat     Chat     Chat     Search       │
│   Cherry    Cherry   Cherry   Cherry   Cherry       │
└─────────────────────────────────────────────────────┘

     "Would you like a slice, sir?"
     "But I wanted the whole cake..."
     "We don't do that here."
```

### What's Wrong with Slices?

1. **Single Flavor Lock-in** - You're stuck with one model's biases and limitations
2. **Rationed Icing** - Your context is afterthought, not architecture
3. **Stale Cherry** - Same interface for everyone, take it or leave it
4. **No Mixing** - Can't combine the chocolate depth with the fruit lightness
5. **Vendor Kitchen** - They bake it, you eat it, no customization

---

## The Solution: The Supercake

What if instead of slices, you got the whole bakery?

```
┌─────────────────────────────────────────────────────────────────┐
│                        THE SUPERCAKE                             │
│                                                                  │
│                    🍒🍒🍒 INTERFACE LAYER 🍒🍒🍒                 │
│              Fresh Cherry │ Glazed │ Candied │ Compote           │
│         (3D Immersive │ Voice │ Chat │ AR/VR │ API)             │
│                                                                  │
│    ═══════════════════════════════════════════════════════════  │
│                                                                  │
│                 🧁 MARZIPAN ONTOLOGY LAYER 🧁                    │
│              Semantic Rules │ OWL Reasoning │ HNSW               │
│           (Your domain's grammar, auto-inferred)                 │
│                                                                  │
│    ═══════════════════════════════════════════════════════════  │
│                                                                  │
│              ════════════════════════════════════                │
│              ║    THICK CONTEXT ICING         ║                  │
│              ║  Knowledge Graph │ RAG │ Memory ║                  │
│              ║  User Data │ Corporate Content  ║                  │
│              ════════════════════════════════════                │
│                                                                  │
│    ═══════════════════════════════════════════════════════════  │
│                                                                  │
│     ┌─────────┬─────────┬─────────┬─────────┬─────────┐         │
│     │ CLAUDE  │ GEMINI  │ DEEPSEEK│PERPLEXITY│ LOCAL  │         │
│     │Fruitcake│Mille-   │Baumkuchen│Sacher- │Sourdough│         │
│     │         │feuille  │         │torte    │ Cake   │         │
│     └─────────┴─────────┴─────────┴─────────┴─────────┘         │
│                    MULTI-MODEL FOUNDATION                        │
│              (All flavors, intelligently routed)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Secret Ingredient: The Marzipan Ontology Layer

Between the icing and the cherry sits something most AI services lack entirely: **the marzipan layer**.

In baking, marzipan creates a smooth foundation for icing, adds flavor complexity, and structurally binds the cake together.

In AI, the ontology layer does the same:

### What Marzipan Does

| Marzipan Function | AI Equivalent |
|-------------------|---------------|
| Smooths the surface | Normalizes data across sources |
| Adds almond richness | Adds semantic meaning to raw facts |
| Structural binding | Enforces logical consistency |
| Prevents icing bleed | Isolates user context properly |
| Enables decoration | Powers intelligent visualization |

### Without Marzipan

```
Your question: "Show me all employees in the London office"

Generic AI: *hallucinates some names*
RAG AI: *retrieves documents, still might hallucinate*
```

### With Marzipan (Ontology)

```
Your question: "Show me all employees in the London office"

Ontology-backed AI:
├── Understands "employee" → Person with Employment relationship
├── Understands "London office" → Location entity with address constraints
├── Infers: If Person worksAt Company, and Company hasOffice London
│           → Person is "employee in London office"
├── Validates: Cannot return Company entities (they're not employees)
└── Returns: Only valid instances with full reasoning chain
```

**The marzipan layer is the grammar checker for your AI's understanding.**

---

## Supercake Capabilities

### 1. Flavor Mixing (Multi-Model)

```
Task: "Research this topic, then write code, then review it"

Supercake Routing:
├── Research: Perplexity (Sachertorte - citation-rich)
├── Planning: DeepSeek (Baumkuchen - layered reasoning)
├── Coding: Claude (Fruitcake - dense & reliable)
└── Review: Gemini (Mille-feuille - multi-perspective)

Cost: 70% less than using premium model for everything
Quality: Best model for each task
```

### 2. Rich Icing (Full Context)

```
Traditional:
┌────────────────┐
│ "You are a    │
│ helpful       │  ← Thin icing (50 tokens)
│ assistant"    │
└────────────────┘

Supercake:
┌────────────────────────────────────────┐
│ 1,475 Knowledge Graph Nodes            │
│ 34,988 Semantic Relationships          │
│ 27 Domain Ontologies                   │  ← Thick icing
│ User History & Preferences             │     (millions of
│ Corporate Policy Constraints           │      tokens of
│ Real-time Context Updates              │      context)
└────────────────────────────────────────┘
```

### 3. Cherry Bar (Interface Choice)

| Your Preference | Interface Available |
|----------------|---------------------|
| Quick chat | Text box |
| Deep exploration | 3D knowledge graph |
| Hands-free | Voice commands |
| Immersive | VR headset (Quest 3) |
| Collaborative | Shared spatial workspace |
| Creative | Media studio with generation |
| API integration | REST/WebSocket/MCP |

### 4. Local Sourdough (Self-Hosted)

```
Cloud AI:
├── Your data goes to their kitchen
├── They see your recipes
├── They control the oven temperature
└── You pay per slice

Self-Hosted (Local LLM):
├── Your kitchen, your rules
├── Your secret ingredients stay secret
├── Full control over everything
└── Fixed cost, unlimited slices
```

---

## The Complete Supercake Recipe

### Ingredients

```
FOUNDATION (The Cakes):
├── 2 cups Claude (primary reasoning)
├── 1 cup DeepSeek (chain-of-thought)
├── 1 cup Perplexity (real-time search)
├── 1/2 cup Gemini (long-context tasks)
├── Optional: Local LLM to taste

MARZIPAN (Ontology):
├── OWL 2 EL semantic rules
├── Domain-specific axioms
├── Inference engine (Whelk-rs)
└── Consistency validation

ICING (Context):
├── Neo4j knowledge graph
├── HNSW vector embeddings
├── User preferences
├── Corporate content
└── Real-time updates

CHERRY COMPOTE (Interfaces):
├── 3D graph visualization
├── Voice interaction
├── VR/AR immersive mode
├── Chat interface
├── API endpoints
└── Media generation tools
```

### Method

1. **Layer the cakes** - Connect multiple models with intelligent routing
2. **Apply marzipan** - Define your domain ontology, let it smooth the surface
3. **Spread thick icing** - Load your knowledge graph, rich and generous
4. **Top with cherries** - Choose your interface(s), fresh to order
5. **Serve complete** - Not a slice, the whole magnificent thing

---

## Comparison: Slice vs Supercake

| Aspect | Industry Slice | Supercake |
|--------|---------------|-----------|
| Foundation | Single model | Multi-model, intelligently routed |
| Context | System prompt | Full knowledge graph |
| Semantics | None | OWL ontology layer |
| Interface | Chat only | 3D, voice, VR, chat, API, media |
| Control | Vendor-controlled | Self-sovereign option |
| Cost | Pay per token | Optimized routing, local option |
| Customization | Minimal | Fully configurable |
| Collaboration | Single user | 50+ concurrent, spatial |

---

## Final Thought

The AI industry has convinced us that a slice of cake is a complete meal.

Meanwhile, we've built a full patisserie where:
- Every flavor is available
- The icing is as thick as you need
- The marzipan ensures structural integrity
- The cherry is fresh, not from a jar
- And yes, you can make your own sourdough version at home

**Stop accepting slices. Demand the whole cake.**

---

*Next: See this analogy visualized in the IRIS Infographic*

plementing in 2026. And one of those
is this idea of context graphs. The
concept is about what it takes to get
agents to do more and more important
work, which it turns out some argues is
not just about giving them access to
better organized data, but about giving
them access to a type of data which
right now might not exist. So today
we're going to talk about what contacts
are and why everyone's talking about it.
But we actually have to go back to an
essay from earlier in December by
investor Jamine Ball who wrote long live
systems of record. And while his
starting point was a debate in the
startup and VC world around how agents
change systems of record and what it
means for the startup landscape,
underlying this is actually a much more
important conversation about how AI is
going to intersect with human knowledge
work. The animating idea that serves as
our jumping off point comes when Ball
writes, "If an enterprise workflow needs
to know something at a specific step,
where is the one place that answer is
considered canonical?" Because as
workflows get more automated and more
agent-driven, the fragility point often
has nothing to do with the model and
everything to do with whether the agent
pulled the right value from the right
system at the right time. Now, Ball goes
on to dramatize this. He writes, "Anyone
who has spent time inside a large
company knows how messy this gets in
practice. Take something as simple as
what is our ARR? Ask the sales or and
you will get one number. Ask finance and
you get another with a different set of
exclusions and adjustments. Ask
accounting and now you're talking
revenue recognition, not bookings. Ask
legal and they will correctly remind you
that half the ARR in a fast growing
business is backed by contracts that
look nothing like the neat recurring
subscriptions you want it to be. Now he
continues, imagine you tell an agent, go
calculate ARR by segment and send a deck
to the board. Which ARR should it use?
Which table is canonical? If sales and
finance disagree, who wins? If the
billing system and the warehouse have
drifted by a few%, which one does the
agent treat as truth? The more we
automate, the more important it becomes
that someone has done the unglamorous
work of deciding what the correct answer
is and where it lives. Now, Paul points
out that even before agents,
rationalizing and reconciling all this
data is something that has been a
priority for big companies over the last
decade. It's where we got all of these
data lakehouse and data warehouse
companies. However, in practice, how
much these things actually change the
way companies operate is up for debate.
Ball writes, "The problem is that most
of this lived downstream of the
operational world. The sales team still
lived in Salesforce. The finance team
still closed the books in Netswuite. The
support team still worked tickets in
Zenesk. The warehouse or lakehouse was
the retrospective mirror, not the
transactional front door. Now agents
come along and in his words change the
equation. Agents are inherently
cross-system, meaning they don't live
solely within one of those functions and
they are actionoriented. They are not
just trying to gather information. They
are trying to make use of that
information to do things. That
combination, he writes, means agents are
only as good as their understanding of
which system owns which truth and what
the contract is between those truths.
Under the hood, something still has to
say this is the canonical customer
record or this is the legally binding
contract term or this number is the one
we report to Wall Street. That something
might be a traditional system of record.
It might be a warehouse back semantic
layer or it might be a new class of data
control plane product but it is
absolutely not going away.
Now this is where we jump from the
systems of record essay into the context
graph idea. This was written again by
investors Jay Agupta and Ashug from
Foundation Capital which they called
AI's trillion dollar opportunity context
graphs. Their key observation is that
there is actually an entire category of
information missing. They write, "Balls
framing assumes the data agents need
already lives somewhere, and agents just
need better access to it, plus better
governance, semantic contracts, and
explicit rules about which definition
wins for which purpose." That's half the
picture. The other half is the missing
layer that actually runs enterprises.
The decision traces, the exceptions,
overrides, precedents, and cross-system
context that currently lives in Slack
threads, deal conversations, escalation
calls, and people's heads. The
distinction, the authors say, is between
rules and decision traces. Rules, they
say, tell an agent what should happen in
general, whereas decision traces capture
what happened in this specific case. I
think for our purposes on this episode,
an even simpler way to understand it is
the what versus why gap. And the simple
idea here is that while systems of
record are good at state, i.e. this
particular deal closed at a 20%
discount, they are bad at decision
lineage. Why a 20% discount was allowed
this time? As the authors point out,
those decision traces, i.e. The why
lives in Slack and DMs and in meetings
and in human heads, limiting how much
autonomy can then scale. As the authors
put it, agents don't just need rules.
They need access to the decision traces
that show how rules were applied in the
past, where exceptions were granted, how
conflicts were resolved, who approved
what, and which precedents actually
govern reality. Now, the good news, they
say, is that agents have a really great
ability to collect exactly this sort of
information. They write system of agent
startups sit in the execution path. They
see the full context at decision time.
What inputs were gathered across
systems? What policy was evaluated? What
exception route was invoked? Who
approved and what state was written? If
you persist those traces, you get
something that doesn't exist in most
enterprises today. A queryable record of
how decisions were made. And this is
what they call the context graph. The
sum total of those decision traces. as
they put it, a living record of decision
traces stitched across entities in time.
So precedent becomes searchable over
time. The context graph becomes the real
source of truth for autonomy because it
explains not just what happened but why
it was allowed to happen. Again, the
what versus why. So just to make this
crisp, here are a couple other examples
that they give. One category is
exception logic that lives in people's
heads. For example, we always give
healthcare companies an extra 10%
because their procurement cycles are
brutal. That's not in the CRM. They
point out it's tribal knowledge passed
down through onboarding and side
conversations. Another category is
precedent from past decisions. We
structured a similar deal for company X
last quarter. We should be consistent.
Again, this is the common knowledge of
the organization that lives in
conversations, not queriable databases.
Another obvious but important one is
cross-system synthesis where a person a
human looks across data in Salesforce
open escalations in Zenesk read a slack
thread where someone flagged churn risk
and ultimately synthesized all of that
in their head decided to escalate
leaving a record that only says
escalated to tier 3. The final category
of examples they give are approval
chains that happen outside of structured
systems. And this happens all the time.
your boss happens to be walking by and
you ask them if you can add an
additional 5% to the discount. They give
a thumbs up and keep walking. The record
is only going to show the final price,
not who approved the deviation or why.
The context graph is the sum total of
all those decision traces if and as they
get captured. So what would it look like
to actually have this context graph
available and play out in real life?
Here's the example they give. A renewal
agent proposes a 20% discount. policy
caps renewals at 10% unless a service
impact exemption is approved. The agent
pulls three SEV1 incidents from pager
duty, an open cancel unless fixed
escalation in Zenesk, and the prior
renewal thread where a VP approved a
similar exemption last quarter. It
routes the exception to finance, finance
approves. The CRM ends up with one fact,
20% discount. The context graph,
however, has all of that information
about why it contains all of those
decision traces. And once you have the
decision records they write, the why
becomes first class data. Over time,
these records naturally form a context
graph. The entities the business already
cares about, counts, renewals, tickets,
incidents, policies, approvers, and
agent runs connected by decision events,
the moments that matter, and why links.
Companies can now audit and debug
autonomy and turn exceptions into
precedent instead of relearning the same
edge cases in Slack every quarter. The
feedback loop they conclude is what
makes this compound. captured decision
traces become searchable precedent and
every automated decision adds another
trace to the graph. Now the big question
quite obviously becomes how do you start
to map this? Is this something that can
only be forward-looking or is there any
way to go backwards? Is this something
that is only going to be relevant as
agents come online which can naturally
create this context graph and map the
why or are we talking about adding the
mapping of decision traces to the human
processes that exist right now. For
example, are you asking leaders to talk
to a voice agent after making decisions
that does that capturing and
categorizing? Now, this essay was hugely
resonant and tons and tons of people
took and ran with the ideas. And one of
the areas where people spent the most
time is in this question of how best to
design these systems. One of the most
interesting follow-ups came from the
cogent enterprise substack who basically
argued to not pre-onstrain the AI in the
design of these context graphs. And this
a little bit hearkens back to the idea
that I've shared before of why I think
automating existing human workflows is
sort of a dead end or at least is mostly
about short-term value. My argument is
that ultimately agents are going to find
ways to do things that come to the same
output differently than the way that
humans would do it. And so trying to
constrain agents to just doing the exact
same things that human did doesn't
really make sense. The cogent enterprise
substack is arguing something similar
about how we think about the design of
context graph mapping. They write the
most counterintuitive development. We
shouldn't predefine these context
graphs. Traditional knowledge graphs
fail because they require predefining
structure up front. Context graphs
invert this completely. Modern agents
act as informed walkers through your
decision landscape. As an agent solves a
problem, traversing through APIs,
querying documentation, reviewing past
tickets, it discovers the organizational
oncology on the fly, it learns which
entities actually matter and how they
genuinely relate through use, not
through a manual schema someone designed
in a workshop. Each trajectory leaves a
trace, which systems were touched
together, which data points co-occurred
in decision chains, how conflicts were
resolved. Accumulate thousands of these
walks and something remarkable emerges.
The organizational schema reveals itself
from actual usage patterns rather than
predetermined assumptions. These become
world models, not just retrieval
systems. Trying to give this a specific
example, you can imagine in a lot of
organizations that they have one type of
policy that in practice they break
almost every time. In other words, the
exception isn't actually the exception.
It's just the rule, but for whatever
reason, it's not been codified into
policy. Going back to that discounting
for health companies example from the
original piece, if that happens every
time, that's actually not an exception.
That's just the policy in practice when
it comes to that type of organization.
And what Cogjent Enterprise is arguing
is that if you allow the agents to
figure out this organizational schema
from the actual real life experience,
they're going to be able to show those
policy in practice areas rather than be
constrained by the preassumptions of
policy that people would have programmed
into them. Now, of course, the other
question that comes up is where humans
fit in all of this. Box's Aaron Levy
also wrote an essay about similar themes
called the era of context. One of the
big questions that he explored was that
if everyone has the same access to
talent, i.e. all of these agentic super
intelligences, how do they then
differentiate? What makes a difference
between a good company and a great
company? For him, it comes back to
context and how we design for it. And
here Aaron puts crisply why context
engineering was one of my key
predictions for enterprise AI in 2026.
He writes, "Designing our systems to get
agents access to that data and ensuring
that all of our agents can interoperate
on that data is going to be incredibly
important. Further, companies will have
to drive a substantial amount of change
management to make this all work. We
imagined that AI systems would adapt to
how we work. But it turns out due to
their extreme power and inherent
limitations, we will instead adapt to
how they work. This means we will have
to optimize our organizations and
workflows to best enable context for
agents to be successful. The core tenant
of this change is that the user is
responsible now for directing and
guiding agents on how to do their work,
ensuring it gets the right context along
the way. In essence, he writes, the
individual contributor of today becomes
the manager of agents in the future.
Their new responsibilities will be
providing the oversight and escalation
paths, a meaningful amount of
coordination throughout the work that
the agents are doing, and shephering
work between the various agents. Just
like managers of teams in the preAI era,
one might say that the decision traces
that make up the context graph are the
most uniquely human part of how work
gets done. They are the decisions that
break the rules or even if they don't
break the rules technically, break out
of the patterns by which previous
decisions were made. So much of being a
good company is about being nimble and
responding to reality as it presents
itself, not as you imagined it. And it
seems to me that as we figure out and
negotiate the relationship between agent
workers and human workers, it's likely
that a lot of the human roles are going
to be in these areas of judgment.
Anyways guys, that is a little primer on
context graphs. I think it's a concept
that you're going to hear a lot more
about this year as part of the larger
conversation around context engineering
and hopefully you now feel more
prepared. For now, that is going to do
it for today's AI daily brief.
Appreciate you listening or watching as

2026: This is AGI
Saddle up: your dreams for 2030 just became possible for 2026.
By Pat Grady and Sonya Huang
Years ago, some leading researchers told us that their objective was AGI. Eager to hear a coherent definition, we naively asked “how do you define AGI?”. They paused, looked at each other tentatively, and then offered up what’s since become something of a mantra in the field of AI: “well, we each kind of have our own definitions, but we’ll know it when we see it.”
This vignette typifies our quest for a concrete definition of AGI. It has proven elusive.
While the definition is elusive, the reality is not. AGI is here, now.
Coding agents
 are the first example. There are more on the way.
Long-horizon agents are functionally AGI, and 2026 will be their year.
Blissfully Unencumbered by the Details
Before we go any further, it’s worth acknowledging that we do not have the moral authority to propose a technical definition of AGI.
We are investors. We study markets, founders, and the collision thereof: businesses.
Given that, ours is a functional definition, not a technical definition. New technical capabilities beg the Don Valentine question: so what?
The answer resides in real world impact.
A Functional Definition of AGI
AGI is the ability to figure things out. That’s it.*
* We appreciate that such an imprecise definition will not settle any philosophical debates.  Pragmatically speaking, what do you want if you’re trying to get something done? An AI that can just figure stuff out. How it happens is of less concern than the fact that it happens.
A human who can figure things out has some baseline knowledge, the ability to reason over that knowledge, and the ability to iterate their way to the answer.
An AI that can figure things out has some baseline knowledge (pre-training), the ability to reason over that knowledge (inference-time compute), and the ability to iterate its way to the answer (long-horizon agents).
The first ingredient (knowledge / pre-training) is what fueled the original ChatGPT moment in 2022. The second (reasoning / inference-time compute) came with the release of o1 in late 2024. The third (iteration / long-horizon agents) came in the last few weeks with Claude Code and other coding agents crossing a capability threshold.
Generally intelligent people can work autonomously for hours at a time, making and fixing their mistakes and figuring out what to do next without being told. Generally intelligent agents can do the same thing. This is new.
What does it mean to figure things out?
A founder messages his agent: "I need a developer relations lead. Someone technical enough to earn respect from senior engineers, but who actually enjoys being on Twitter. We sell to platform teams. Go."
The agent starts with the obvious: LinkedIn searches for "Developer Advocate" and "DevRel" at great developer-first companies — Datadog, Temporal, Langchain. It finds hundreds of profiles. But job titles don't reveal who's actually good at this.
It pivots to signal over credentials. It searches YouTube for conference talks. It finds 50+ speakers, then filters for those with talks that have strong engagement.
It cross-references those speakers with Twitter. Half have inactive accounts or just retweet their employer's blog posts. Not what we want. But a dozen have real followings — they post real opinions, reply to people, and get engagement from developers. And their posts have real taste.
The agent narrows further. It checks who's been posting less frequently in the last three months. A drop in activity sometimes signals disengagement from their current role. Three names surface.
It researches those three. One just announced a new role — too late. One is a founder of a company that just raised funding — not leaving. The third is a senior DevRel at a Series D company that just did layoffs in marketing. Her last talk was about exactly the platform engineering space the startup targets. She has 14k Twitter followers and posts memes that actual engineers engage with. She hasn't updated her LinkedIn in two months.
The agent drafts an email acknowledging her recent talk, the overlap with the startup's ICP, and a specific note about the creative freedom a smaller team offers. It suggests a casual conversation, not a pitch.
Total time: 31 minutes. The founder has a shortlist of one instead of a JD posted to a job board.
This is what it means to figure things out. Navigating ambiguity to accomplish a goal – forming hypotheses, testing them, hitting dead ends, and pivoting until something clicks. The agent didn't follow a script. It ran the same loop a great recruiter runs in their head, except it did it tirelessly in 31 minutes, without being told how.
To be clear: agents still fail. They hallucinate, lose context, and sometimes charge confidently down exactly the wrong path. But the trajectory is unmistakable, and the failures are increasingly fixable.
How did we get here? From reasoning models to long-horizon agents
In last year’s essay, we wrote
 about reasoning models as the most important new frontier for AI. Long-horizon agents push this paradigm further by allowing models to take actions and iterate over time.
Coaxing a model to think for longer is not trivial. A base reasoning model can think for seconds or minutes.
Two different technical approaches seem to both be working and scaling well: reinforcement learning and agent harnesses. The former approach teaches a model intrinsically to stay on track for longer by poking and prodding it to maintain focus during the training process. The latter designs specific scaffolding around the known limitations of models (memory hand-offs, compaction, and more).
Scaling reinforcement learning is the domain of the research labs. They have made exceptional progress on this front, from multi-agent systems to reliable tool use.
Designing great agent harnesses is the domain of the application layer. Some of the most beloved products on the market today are known for their exceptionally engineered agent harnesses: Manus, Claude Code, Factory’s Droids, etc.
If there’s one exponential curve to bet on, it’s the performance of long-horizon agents. METR has been meticulously tracking
 AI’s ability to complete long-horizon tasks. The rate of progress is exponential, doubling every ~7 months. If we trace out the exponential, agents should be able to work reliably to complete tasks that take human experts a full day by 2028, a full year by 2034, and a full century by 2037.
So What?
Soon you’ll be able to hire an agent. That’s one litmus test for AGI (h/t: Sarah Guo).
You can “hire” GPT-5.2 or Claude or Grok or Gemini today. More examples are on the way:

    Medicine: OpenEvidence’s Deep Consult functions as a specialist
    Law: Harvey’s agents function as an Associate
    Cybersecurity: XBOW functions as a pen-tester
    DevOps: Traversal’s agents function as an SRE
    GTM: Day AI functions as a BDR, SE, and Rev Ops leader
    Recruiting: Juicebox functions as a recruiter
    Math: Harmonic’s Aristotle functions as a mathematician
    Semiconductor Design: Ricursive's agents function as chip designers
    AI Researcher: GPT-5.2 and Claude function as AI researchers

From Talkers to Doers: Implications for Founders
This has profound implications for founders.
The AI applications of 2023 and 2024 were talkers. Some were very sophisticated conversationalists! But their impact was limited.
The AI applications of 2026 and 2027 will be doers. They will feel like colleagues. Usage will go from a few times a day to all-day, every day, with multiple instances running in parallel. Users won’t save a few hours here and there – they’ll go from working as an IC to managing a team of agents.
Remember all that talk of selling work? Now it’s possible.
What work can you accomplish? The capabilities of a long-horizon agent are drastically different than a single forward pass of a model. What new capabilities do long-horizon agents unlock in your domain? What tasks require persistence, where sustained attention is the bottleneck?
How will you productize that work? How will your application interface evolve in your domain, as the UI of work grows from chatbot to agent delegation?
Can you do that work reliably? Are you obsessively improving your agent harness? Do you have a strong feedback loop?
How can you sell that work? Can you price and package to value and outcomes?
Saddle Up!
It’s time to ride the long-horizon agent exponential.
Today, your agents can probably work reliably for ~30 minutes. But they’ll be able to perform a day’s worth of work very soon – and a century’s worth of work eventually.
What can you achieve when your plans are measured in centuries? A century is 200,000 clinical trials no one's cross-referenced. A century is every customer support ticket ever filed, finally mined for signal. A century is the entire U.S. tax code, refactored for coherence.
The ambitious version of your roadmap just became the realistic one.
—
Published on January 14, 2026
Thanks to Dan Roberts, Harrison Chase, Noam Brown, Sholto Douglas, Isa Fulford, Ben Mann, Nick Turley, Phil Duan, Michelle Bailhe, and Romie Boyd for reviewing drafts of this post.

Toward a Definition of AGI
When an infant is born, they are completely dependent on their caregivers to survive. They can’t eat, move, or play on their own. As they grow, they learn to tolerate increasingly longer separations.
Gradually, the caregiver occasionally and intentionally fails to meet their needs: The baby cries in their crib at night, but the parent waits to see if they’ll self-soothe. The toddler wants attention, but the parent is on the phone. These small, manageable disappointments—what the psychologist D.W. Winnicott called "good-enough parenting"
—teach the child that they can survive brief periods of independence.
Over months and years, these periods extend from seconds to minutes to hours, until eventually the child is able to function independently.
AI is following the same pattern.
Today we treat AI like a static tool we pick up when needed and set aside when done. We turn it on for specific tasks—writing an email, analyzing data, answering questions—then close the tab. But as these systems become more capable, we'll find ourselves returning to them more frequently, keeping sessions open longer, and trusting them with more continuous workflows. We already are.
So here’s my definition of AGI
:
AGI (artificial general intelligence) is achieved when it makes economic sense to keep your agent running continuously.
In other words, we’ll have AGI when we have persistent agents that continue thinking, learning, and acting autonomously between your interactions with them—like a human being does.
I like this definition because it’s empirically observable: Either people decide it’s better to never turn off their agents or they don’t. It avoids the philosophical rigmarole inherent to trying to define what true general intelligence is. And it avoids the problems of the Turing Test and OpenAI’s definition of AGI.
In the Turing Test, a system is AGI when it can fool a human judge into thinking it is human. The problem with the Turing Test is that it sets up moveable goalposts: If I interacted with GPT-4 10 years ago, I would have thought it was human. Today, I’d simply ask it to build a website for me from scratch, and I’d know instantly it was not human.
OpenAI’s definition of AGI—which is AI that can outperform humans at most economically valuable work—suffers from the same problem. What constitutes economically valuable work constantly changes. We will invent new economically valuable work that we perform in conjunction with AI. These hybrid roles then become the new benchmark that AI will need to learn to do before it counts as AGI. So the definition is an ever-receding target.
By contrast, the definition I proposed—AGI is achieved when it makes economic sense to keep your agent running continuously—is a binary, irreversible, and immovable threshold: Once we are running our agents 24/7, we’ve hit it, and there’s no going back. (After all, we can’t uninvent it.)
I like this definition because in order to meet it we will need to develop a lot of necessary but hard-to-define components of AGI:

    Continuous learning: The agent must learn from experience without explicit user prompting.
    Memory management: The agent needs sophisticated ways to store, retrieve, and forget information efficiently over extended periods.
    Generating, exploring, and achieving goals: The agent requires the open-ended ability to define new, useful goals and maintain them across days, weeks, or months, while adapting to changing circumstances.
    Proactive communication: The agent should reach out when it has updates, questions, or requires input, rather than only responding when summoned. It must also be able to be interrupted and redirected by the user.
    Trust and reliability: The agent must be safe and reliable. Users will not keep agents running unless they are confident the system will not cause harm or make costly errors autonomously.

While I've described these capabilities, I'm deliberately avoiding the trap of trying to specify exact technical criteria for each one. What precisely constitutes “continuous learning” or “trust” is difficult to pin down.
Instead, my AGI definition entails that all of these capabilities are present to some extent
And these capabilities already are present in limited ways: ChatGPT, for example, has rudimentary forms of memory
 and proactive communication.
The length of time during which AI can run on its own is increasing gradually and consistently. When GPT-3 first came out, the primary use case for AI was the GitHub Copilot—the best it could do was complete the line of code you were already writing.
ChatGPT lengthened the amount of time the AI could run from the amount required for you to press “tab” to complete a line of code to the time required to deliver a full response in a chat conversation. Now, agentic tools like Claude Code
, deep research
, and Codex
 can run for between 5-20 minutes at a stretch.
The trajectory is clear: from seconds to minutes to hours, and to days and beyond.
Eventually, the cognitive and economic costs of starting fresh each time will outweigh the benefits of turning AI off.