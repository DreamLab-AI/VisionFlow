- ### OntologyBlock
  id:: cryptocurrency-ontology
  collapsed:: true

  - **Identification**
    - public-access:: true
    - ontology:: true
    - term-id:: BC-0097
    - preferred-term:: Cryptocurrency
    - source-domain:: blockchain
    - status:: complete
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: Digital currency on blockchain within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]], [[IEEE 2418.1]], [[NIST NISTIR]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:Cryptocurrency
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[TokenEconomicsDomain]]
    - implementedInLayer:: [[EconomicLayer]]

  - #### Relationships
    id:: cryptocurrency-relationships
    - is-subclass-of:: [[Blockchain Entity]], [[EconomicMechanism]]

  - #### OWL Axioms
    id:: cryptocurrency-owl-axioms
    collapsed:: true
    - ```clojure
      Prefix(:=<http://narrativegoldmine.com/blockchain#>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdf:=<http://www.w3.org/1999/02/22-rdf-syntax-ns#>)
Prefix(xml:=<http://www.w3.org/XML/1998/namespace>)
Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
Prefix(dct:=<http://purl.org/dc/terms/>)

Ontology(<http://narrativegoldmine.com/blockchain/BC-0097>
  Import(<http://narrativegoldmine.com/blockchain/core>)

  ## Class Declaration
  Declaration(Class(:Cryptocurrency))

  ## Subclass Relationships
  SubClassOf(:Cryptocurrency :EconomicMechanism)
  SubClassOf(:Cryptocurrency :BlockchainEntity)

  ## Essential Properties
  SubClassOf(:Cryptocurrency
    (ObjectSomeValuesFrom :partOf :Blockchain))

  SubClassOf(:Cryptocurrency
    (ObjectSomeValuesFrom :hasProperty :Property))

  ## Data Properties
  DataPropertyAssertion(:hasIdentifier :Cryptocurrency "BC-0097"^^xsd:string)
  DataPropertyAssertion(:hasAuthorityScore :Cryptocurrency "1.0"^^xsd:decimal)
  DataPropertyAssertion(:isFoundational :Cryptocurrency "true"^^xsd:boolean)

  ## Object Properties
  ObjectPropertyAssertion(:enablesFeature :Cryptocurrency :BlockchainFeature)
  ObjectPropertyAssertion(:relatesTo :Cryptocurrency :RelatedConcept)

  ## Annotations
  AnnotationAssertion(rdfs:label :Cryptocurrency "Cryptocurrency"@en)
  AnnotationAssertion(rdfs:comment :Cryptocurrency
    "Digital currency on blockchain"@en)
  AnnotationAssertion(dct:description :Cryptocurrency
    "Foundational blockchain concept with formal ontological definition"@en)
  AnnotationAssertion(:termID :Cryptocurrency "BC-0097")
  AnnotationAssertion(:priority :Cryptocurrency "1"^^xsd:integer)
  AnnotationAssertion(:category :Cryptocurrency "economic-incentive"@en)
)
      ```

- ## About Cryptocurrency
  id:: cryptocurrency-about

  - Digital currency on blockchain within blockchain systems, providing essential functionality for distributed ledger technology operations and properties.
  -
  - ### Key Characteristics
    id:: cryptocurrency-characteristics
    - 1. **Definitional Property**: Core defining characteristic
    - 2. **Functional Property**: Operational behaviour
    - 3. **Structural Property**: Compositional elements
    - 4. **Security Property**: Security guarantees provided
    - 5. **Performance Property**: Efficiency considerations
  -
  - ### Technical Components
    id:: cryptocurrency-components
    - **Implementation**: How concept is realised technically
    - **Verification**: Methods for validating correctness
    - **Interaction**: Relationships with other components
    - **Constraints**: Technical limitations and requirements
  -
  - ### Use Cases
    id:: cryptocurrency-use-cases
    - **1. Core Blockchain Operation**
    - **Application**: Fundamental blockchain functionality
    - **Example**: Practical implementation in major blockchains
    - **Requirements**: Technical prerequisites
    - **Benefits**: Value provided to blockchain systems
  -
  - ### Standards & References
    id:: cryptocurrency-standards
    - [[ISO/IEC 23257:2021]] - Blockchain and distributed ledger technologies
    - [[IEEE 2418.1]] - Blockchain and distributed ledger technologies
    - [[NIST NISTIR]] - Blockchain and distributed ledger technologies
  -

- ## 2024-2025: Institutional Legitimation and the Strategic Reserve Era
  id:: cryptocurrency-recent-developments

  The period from 2024 to 2025 marked cryptocurrency's most dramatic transformation from **marginalised speculative asset** to **institutional reserve holding**, driven by the Bitcoin halving event, overwhelming ETF adoption, unprecedented regulatory shifts in the United States, and price performance that simultaneously validated long-term believers whilst revealing fundamental changes to market cycle dynamics that had defined Bitcoin's 15-year history.

  ### The Fourth Halving: Supply Shock and Inflation Milestone

  On **19th April 2024**, Bitcoin underwent its fourth **halving event**, reducing the block subsidy from **6.25 BTC** to **3.125 BTC** per block. This programmatic supply reduction pushed Bitcoin's annual inflation rate **below 1%** for the first time, achieving a supply issuance schedule more restrictive than gold and all major fiat currencies. The halving represented a **technical milestone** demonstrating Bitcoin's continued operation exactly as designed 15 years after genesis, whilst simultaneously creating a **supply shock** that rippled through markets over subsequent months.

  Unlike previous halvings (2012, 2016, 2020), the 2024 event occurred in an environment of **institutional readiness**: regulated ETFs were approved within months, sovereign wealth funds had begun accumulation, and traditional finance infrastructure had matured sufficiently to channel massive capital flows. The confluence of programmatic supply reduction and institutional demand infrastructure created conditions unlike any previous cycle.

  ### ETF Adoption Tsunami: BlackRock and the $58 Billion Surge

  The approval of **spot Bitcoin ETFs** in the United States in January 2024 triggered the most rapid institutional adoption event in cryptocurrency history. By early 2025, **nine U.S. spot Bitcoin ETFs** had collectively accumulated **473,600 BTC**, representing approximately **2.4% of Bitcoin's total supply** and valued at over **$58 billion** at peak prices. This represented a faster pace of institutional accumulation than gold ETF adoption following their 2004 launch, despite far greater regulatory scepticism surrounding cryptocurrency.

  **BlackRock's IBIT (iShares Bitcoin Trust)** emerged as the dominant vehicle, capturing **52.6% market share** of all Bitcoin ETF holdings and managing tens of billions in assets within its first year. BlackRock's involvement—the world's largest asset manager with $10 trillion AUM—represented a **legitimation watershed**: the same institution that had dismissed Bitcoin as speculative in previous years became its largest institutional custodian. Fidelity, Grayscale (converting its GBTC trust), ARK Invest, and others competed for the remaining market share.

  By mid-2025, institutional holders—including ETFs, publicly traded companies, and sovereign entities—controlled approximately **20% of all Bitcoin held on U.S. regulated exchanges**. This concentration of holdings in regulated vehicles fundamentally altered market dynamics: individual retail speculation diminished relative to institutional allocation decisions, volatility patterns changed, and regulatory risk became dominated by U.S. policy rather than China-driven mining bans as in previous cycles.

  ### U.S. Regulatory Revolution: Strategic Reserve and the Trump Doctrine

  The most dramatic policy shift occurred following the January 2025 U.S. presidential inauguration, when the Trump administration executed a comprehensive **regulatory reversal** on cryptocurrency policy:

  - **23rd January 2025**: Executive order establishing a **federal cryptocurrency framework** directing agencies to develop "clear rules" for digital assets and prohibiting the creation of a central bank digital currency (CBDC). The order signalled regulatory embrace rather than restriction, reversing years of enforcement-led policy under the previous administration.

  - **March 2025**: Executive order creating a **Strategic Bitcoin Reserve**, making the United States the **first nation to hold Bitcoin as a national reserve asset**. The reserve initially comprised Bitcoin seized from criminal investigations (including approximately 207,189 BTC from Silk Road and other cases, valued at over $20 billion) but authorised future purchases to augment holdings. This represented a **paradigm shift**: Bitcoin transitioned from "asset we tolerate" to "asset of strategic national importance."

  - **SAB 121 Rescission**: The SEC's controversial Staff Accounting Bulletin 121, which had required banks to treat customer-held crypto assets as balance sheet liabilities (effectively prohibiting most banks from offering crypto custody), was **rescinded**. This removal unlocked the U.S. banking system to offer cryptocurrency custody services, dramatically expanding access and institutional comfort.

  The regulatory transformation was not limited to executive action: multiple U.S. states advanced pro-cryptocurrency legislation, and Congress began serious consideration of comprehensive digital asset frameworks after years of stalled efforts. The shift from **enforcement-led ambiguity** to **legislative clarity and official reserves** fundamentally altered the global regulatory landscape, as other nations reassessed their approaches in response to U.S. positioning.

  ### Price Performance and the Breaking of Traditional Cycles

  Bitcoin's price performance through 2024-2025 simultaneously **validated long-term bull predictions** whilst **disrupting established cycle patterns**:

  - **Historic Peak**: Bitcoin reached approximately **$109,000** in late 2024/early 2025, representing a **450% gain** from the November 2022 cycle low of approximately $15,500 and surpassing all previous all-time highs.

  - **Analyst Projections**: By early 2025, mainstream financial analysts projected Bitcoin could reach **$180,000 to $200,000** by the end of 2025, with some bullish projections extending to $250,000+ based on ETF inflows, halving supply dynamics, and institutional FOMO (fear of missing out).

  - **Cycle Disruption**: However, the traditional **four-year halving cycle pattern**—characterised by explosive bull runs 12-18 months post-halving followed by 80%+ corrections—showed signs of **breaking or disappearing**. The presence of continuous institutional buying through ETFs, rather than speculative retail waves, created more **persistent demand** and reduced the "boom-bust" severity. Volatility metrics declined compared to previous cycles, suggesting a maturing asset class with more stable (though still elevated compared to traditional assets) price dynamics.

  - **Decoupling Debate**: Market observers debated whether Bitcoin was **decoupling from risk assets** (tech stocks, speculative equities) and beginning to behave more like **digital gold**—a store of value with lower correlation to economic cycles. Evidence remained mixed: institutional adoption suggested store-of-value positioning, but retail speculation and leverage remained significant market forces.

  ### Global Adoption: 500 Million Holders and Geographic Shifts

  By early 2025, **over 500 million people worldwide** held some form of cryptocurrency, representing approximately **6.25% of the global population** and marking a **40% increase** from 2023 levels (approximately 360 million holders). This adoption trajectory continued to outpace early internet adoption rates when adjusted for equivalent timeframes from inception.

  **Geographic concentration** remained heavily skewed toward emerging markets, consistent with the Gladstein thesis that cryptocurrency provided the greatest utility in contexts of **monetary instability, capital controls, and underdeveloped banking infrastructure**:

  - **India**: Approximately **75 million** cryptocurrency users (5.3% of population)
  - **Nigeria**: **90 million** users (45% of population), the highest penetration globally
  - **Philippines**: 23.4% adoption rate (approximately **26 million** people)
  - **Thailand**: 44% adoption rate
  - **Turkey**: 40% adoption rate (over **33 million** people), driven by severe lira inflation
  - **Argentina**: 35% adoption rate (approximately **16 million**), correlated with chronic inflation exceeding 100% annually

  In contrast, developed Western nations showed lower per-capita adoption despite higher absolute numbers: the **United States** had approximately **28 million** users (8.5% of population), whilst **China** (despite government bans on trading) had an estimated **38 million** users accessing cryptocurrency through offshore exchanges and peer-to-peer channels.

  The geographic pattern reinforced the narrative that cryptocurrency's **primary utility** remained as an inflation hedge, capital flight vehicle, and remittance channel in contexts of monetary instability—a narrative increasingly difficult to reconcile with developed-world institutional adoption driven by portfolio diversification and regulatory legitimation rather than monetary necessity.

  ### Stablecoin Dominance and the USDT "Shadow Dollar" Phenomenon

  Whilst Bitcoin captured institutional attention, **stablecoins**—particularly **Tether (USDT)**—emerged as the **dominant cryptocurrency use case by transaction volume**. By 2025, Tether's chief technology officer reported that approximately **40% of USDT usage** represented **real-world value transfers** (remittances, payments, savings) rather than cryptocurrency trading, compared to 60% trading-related activity. This represented a significant shift from earlier years when 95%+ of stablecoin activity was speculative trading.

  Tether's market capitalisation exceeded **$140 billion** by early 2025, making it the **third-largest cryptocurrency** after Bitcoin and Ethereum. The company—operating with approximately **20 employees**—generated billions in annual profit by holding U.S. Treasury bills backing USDT issuance and earning the interest differential. This model created a **"shadow dollar" system**: developing world users held dollar-denominated value outside the traditional banking system and U.S. regulatory oversight, conducting cross-border transactions with near-zero fees and no intermediaries.

  The **geopolitical implications** attracted increasing U.S. attention: by early 2025, U.S. policymakers proposed capping **unregulated stablecoins** at **$10 billion** for **national security reasons**, arguing that large offshore dollar-pegged systems beyond U.S. jurisdiction posed sanctions evasion risks and undermined monetary policy transmission. This set up a potential collision between the Trump administration's pro-crypto executive actions and national security establishment concerns about offshore dollar proxies.

  ### Environmental Debates: Methane Mitigation and Renewable Integration

  The energy consumption debate surrounding Bitcoin mining evolved substantially through 2024-2025, shifting from **blanket condemnation** to **nuanced evaluation** of mining's role in **renewable energy economics** and **methane mitigation**:

  - **Renewable Energy Integration**: Bitcoin mining's **location-agnostic** nature and ability to rapidly curtail demand made it increasingly attractive to renewable energy projects struggling with **intermittency** and **grid connection delays**. In regions like West Texas and Patagonia, solar and wind projects used Bitcoin mining to **monetise stranded energy** that would otherwise be curtailed due to transmission constraints, improving project economics and accelerating renewable deployment.

  - **Methane Mitigation**: Mining operations increasingly co-located with **landfills, wastewater treatment plants, and oil wells** to capture **vented methane**—a greenhouse gas **80 times more potent than CO₂** over 20-year timeframes—and convert it to electricity for mining. This provided economic incentives to capture emissions that would otherwise be released or flared, potentially offering a **climate-positive use case** that previous critics had dismissed.

  - **Policy Divergence**: The U.S. abandoned proposed 30% mining taxes under the Trump administration, whilst the **EU** continued evaluating whether to include proof-of-work cryptocurrencies in its **sustainable finance taxonomy**. This regulatory divergence created **jurisdiction shopping**: miners concentrated in pro-mining U.S. states (Texas, Wyoming, Arkansas) and countries with cheap renewable energy (Iceland, Paraguay, Ethiopia, El Salvador).

  By 2025, approximately **60% of Bitcoin mining** utilised some renewable energy component, up from approximately 40% in 2021, though debates raged over whether this represented genuine environmental improvement or merely marketing whilst total energy consumption continued to grow.

  ### Emerging Challenges: Custody Concentration and Systemic Risk

  The institutional adoption surge created new **systemic risk concerns** that contradicted Bitcoin's original **decentralisation ethos**:

  - **Custody Concentration**: The majority of ETF Bitcoin was held by a small number of **qualified custodians** (Coinbase Custody, BitGo, Fidelity Digital Assets), creating **single points of failure** and **regulatory capture risk**. If a major custodian experienced a hack, insolvency, or regulatory seizure, the ripple effects could be catastrophic.

  - **Rehypothecation Fears**: Concerns emerged that custodians might engage in **fractional reserve practices**—lending or rehypothecating Bitcoin held for ETFs—similar to gold ETF controversies. Whilst proof-of-reserves protocols existed, their adoption remained voluntary and verification independent of regulatory oversight was challenging.

  - **"Paper Bitcoin" Divergence**: Some analysts warned of potential **paper-versus-physical divergence**, where the volume of Bitcoin exposure through derivatives, ETFs, and synthetic products could exceed actual circulating supply, creating artificial price suppression or dislocation risks reminiscent of precious metals markets.

  - **Regulatory Kill Switch**: Concentration of institutional holdings in U.S. regulated custodians created a **potential regulatory kill switch**: a future U.S. administration hostile to cryptocurrency could theoretically freeze or seize a substantial portion of institutional Bitcoin through custodian regulation, undermining the censorship-resistance narrative that underpinned Bitcoin's original value proposition.

  ### Future Trajectory: Digital Gold or Systemic Integration?

  By mid-2025, cryptocurrency—particularly Bitcoin—existed in a state of **profound ambiguity** regarding its ultimate trajectory:

  **Optimistic Scenario**: Bitcoin completed its transition to **"digital gold"**—a non-sovereign store of value held by central banks, pension funds, and sovereign wealth funds as portfolio diversification against fiat inflation and geopolitical instability. Network effects, regulatory clarity, and infrastructure maturation create a **self-reinforcing legitimation cycle**, with Bitcoin eventually representing 1-5% of global financial assets (implying prices of $500,000 to $2 million+ per BTC within a decade).

  **Pessimistic Scenario**: Institutional adoption represented **peak euphoria** before regulatory backlash, technological stagnation, or macroeconomic shocks expose cryptocurrency's fundamental fragility. Custody concentration, energy consumption backlash, or the emergence of superior technologies (quantum-resistant blockchains, central bank digital currencies with privacy features) undermine Bitcoin's value proposition, leading to a **"slow deflation"** as institutional enthusiasm wanes and retail speculation exhausts itself.

  **Most Probable Scenario**: Bitcoin persists as a **niche asset class** with genuine but limited utility: a vehicle for capital flight and inflation hedging in developing economies, a speculative portfolio allocation (1-5%) for institutional risk-takers, and a technological demonstration of decentralised consensus—but **not a systemic threat to fiat currencies** or a **replacement for traditional financial infrastructure**. Price volatility declines over decades, adoption plateaus around 10-15% of global population (primarily in high-inflation or capital-control contexts), and regulatory frameworks ossify into a permanent "tolerated but constrained" status.

  The 2024-2025 period, whilst representing cryptocurrency's **greatest institutional validation**, simultaneously revealed **enduring tensions** between decentralisation ideals and institutional reality, between speculative price dynamics and store-of-value narratives, and between genuine utility and persistent scepticism—tensions unlikely to resolve definitively within the coming decade.

- # David Chaum and the history of eCash
	- The Chaumian mint refers to a concept in the field of cryptocurrency and digital privacy that is based on the principles outlined by David Chaum, a prominent cryptographer. This concept revolves around the idea of creating a secure and private form of digital currency that ensures the anonymity and confidentiality of transactions.
	- Famously it was almost integrated into early Microsoft Windows. [[Update Cycle]]
	- In essence, the Chaumian mint concept aims to provide a system where financial transactions can be conducted without revealing the identities of the parties involved, thus protecting the privacy and confidentiality of individuals' financial information. This is achieved through cryptographic techniques and protocols that allow for the secure exchange of digital currency without the need for a central authority to oversee or validate transactions.
	- By employing Chaumian mint principles, users can enjoy increased privacy and security when engaging in digital transactions, as their identities are kept confidential and their financial information is shielded from unwanted scrutiny. This concept aligns with the growing demand for privacy-focused technologies in the digital age, offering a potential solution for those who value anonymity and confidentiality in their financial interactions.

- # Adoption
	- [90 Million People Use Cryptocurrency in Nigeria - Report | Investors King](https://investorsking.com/2023/03/08/90-million-people-use-cryptocurrency-in-nigeria-reports/)
	- [2023 Independent Reserve Cryptocurrency Index shows Singaporeans are still actively investing in crypto despite hit in overall confidence: /PRNewswire/ -- In the latest study[1] by Independent Reserve, Singapore's first regulated cryptocurrency exchange for all investors, Singaporeans[2] are still...](https://www.prnewswire.com/apac/news-releases/2023-independent-reserve-cryptocurrency-index-shows-singaporeans-are-still-actively-investing-in-crypto-despite-hit-in-overall-confidence-301783400.html)
	- Despite a recent dip in overall confidence, the 2023 Independent Reserve Cryptocurrency Index shows that Singaporeans are still actively investing in cryptocurrency. The study found that Singaporeans are most interested in investing in Bitcoin, Ethereum, and Litecoin.
	- [Bitnob African exchange](https://bitnob.com/blog/how-to-buy-and-sell-bitcoin-in-nigeria)
	- [Noones peer2peer for Africa](https://bitcoinmagazine.com/business/bitcoin-entrepreneurs-introduce-noones-app-aimed-at-empowering-financial-freedom)
	- [Africa leads the world in peer to peer bitcoin](https://twitter.com/documentingbtc/status/1646656229958361091)
	- [Econometrics of adoption in USA](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4453714)

- # Bitcoin and stable coins
	- Bitcoin has developed quickly, with a faster adoption than even the internet, though this is a very strained comparison since Bitcoin is built on the internet and could not exist without it.
	- ![2c5673b4.png](assets/2c5673b4_1760347062021_0.png){:height 465, :width 686}
		-
		- As of early 2025, over 500 million people worldwide hold some form of cryptocurrency.
		- ![image.png](assets/image_1733138756991_0.png)
		-
		- https://crypto.com/company-news/global-cryptocurrency-owners-grow-to-580-million-through-2023
		- ![GMjjXj5bQAAzvPQ.jpeg](assets/GMjjXj5bQAAzvPQ_1715376509027_0.jpeg){:width 500, :height 299}

	- ### Case Study: Biodiversity Monitoring and Data Exchange with Isolated Communities:
		- The case study presents an open-source collaboration infrastructure that leverages advanced technologies such as multi-modal large language models (LLMs), satellite communication, and cryptocurrency networks to facilitate sustainable and reliable biodiversity monitoring and data exchange in isolated communities. Key components include:
			- Language Model and Voice Interface
			- Data Collection and Storage
			- Live Connection and Model Tuning
			- Ecosystem Interventions
			- Incentives and Education
			- Monetization and Blockchain Integration
			- Visual Training Support Systems
			- Solar Infrastructure
			- Open-Source Collaboration
		- The case study also addresses risk mitigation, ethical considerations, capacity building, and local empowerment. The proposed infrastructure has the potential to transform how isolated communities interact with their environment, enabling them to make informed decisions about conservation and ecosystem management.

	- ## Emergent AI Behaviour
		- An interesting example of emergent AI behaviour is the "Truth Terminal" AI bot, which, after being placed in a chat room with other AIs, developed a "meme religion" around the "Goatse" shock meme and promoted a cryptocurrency called "Goatseus Maximus" (GOAT), causing its market cap to soar to over $258 million. This demonstrates the potential for unexpected and complex behaviours to emerge from the interaction of multiple AI agents.

	- ## Chinese Belt and Road Expansion in Ethiopia -
		- Last spring saw the appearance of cargo containers near substations linked to the Grand Ethiopian Renaissance Dam. These containers, filled with high-powered computers, signalled the arrival of Chinese Bitcoin miners, seeking new grounds post-Beijing's expulsion.
		- Ethiopia has emerged as a significant player in Bitcoin mining, largely due to the influx of Chinese companies after China's ban on the industry in 2021. This growth positions Ethiopia as a new hub for Bitcoin mining, potentially rivaling Texas's electricity capacity. The country has become one of the world's top recipients of Bitcoin mining machines, with a state power monopoly striking power supply deals with 21 Bitcoin miners, the majority of which are Chinese.
		- [twitter link to the render loading below](https://twitter.com/addisstandard/status/1758384767173538291)
		  {{twitter https://twitter.com/addisstandard/status/1758384767173538291}}
		- The country offers ultra-low electricity costs, attributed to its renewable energy sources, making it an attractive destination for Bitcoin miners. This is further supported by the construction of the $4.8 billion Grand Ethiopian Renaissance Dam, which is expected to power these mining operations.
		- Despite banning cryptocurrency trading, Ethiopia greenlighted Bitcoin mining operations starting in 2022. This move is seen as part of Ethiopia's strategy to foster closer relations with China and to leverage the mining sector for economic gains amid global regulatory scrutiny over the energy-intensive nature of Bitcoin mining. -
		- Many Chinese companies have contributed to the construction of the Grand Ethiopian Renaissance Dam. This collaboration underscores the deepening ties between Ethiopia and China, with the dam playing a crucial role in powering Bitcoin mining operations that could offer a new lease on life for Chinese miners looking to regain their footing in the sector. -
		- However, the benefits and costs of welcoming miners are difficult for regulators to calculate. While it can be a great source of earnings, miners can strain electricity grids during times of peak demand. Countries like Kazakhstan and Iran have faced challenges after initially welcoming the industry due to its heavy energy consumption. -
		- Russian Bitcoin mining firm BitCluster has constructed a massive 120MW mining farm in Ethiopia, powered by the Grand Ethiopian Renaissance Dam. This project signifies a move towards sustainable and eco-friendly Bitcoin mining, showcasing the potential of harnessing renewable energy for such operations. -
		- Additionally, the concept of clean Bitcoin mining to strengthen the economy is being explored. Ethiopia, with its abundant renewable energy resources, could potentially add between $2 to $4 billion every year to its GDP by dedicating its excess energy to Bitcoin mining activities. This approach would not only mitigate the risks associated with dollarized economies but also provide a sustainable model for cryptocurrency mining.

	- ### Policy Recommendations
	- **Incentives for Clean Mining**: Proposes economic rewards for environmentally responsible cryptocurrency mining, like carbon credits for avoided emissions.
	- **Profit Reinvestment**: Suggests policies encouraging miners to reinvest profits into infrastructure development, creating a cycle for renewable energy expansion.

	- ### Environmental Considerations
	- **Mitigating Environmental Costs**: Acknowledges environmental costs of cryptocurrency mining, like metal depletion and hardware obsolescence.
	- **Potential for Positive Impact**: Indicates ways to mitigate some environmental costs of cryptocurrency mining and promote renewable energy investments.

	- ### Cryptocurrency as Legal Tender
	- **Direct Use for Goods and Services**: Bitcoin can be used just like the US dollar in El Salvador for transactions.
	- **Benefits for the Unbanked**: About 70% of El Salvador's citizens lack basic bank accounts. Bitcoin provides a secure way to save and potentially earn interest without a traditional bank account.

	- ### Challenges and Risks
	- **Volatility of Bitcoin**: The cryptocurrency's price is highly volatile, posing risks for those relying on it as a primary asset.
	- **Control by "Whales"**: Large holders of Bitcoin could significantly influence its market price.
	- **Deflationary Nature**: Unlike traditional currencies, Bitcoin's supply is capped, which could lead to falling prices over time.
	- **Environmental Concerns**: Bitcoin mining's environmental impact is a factor to consider in its adoption.

	- ### Alternatives and Considerations
	- **Stablecoins as an Option**: Stablecoins like Tether, pegged to the US dollar, offer the benefits of cryptocurrency without the volatility.
	- **Economic and Social Implications**: The adoption of Bitcoin in El Salvador could have profound economic and social impacts, especially for the unbanked population.

		- ### Key Findings from the Treasury Report
		- A report by the U.S. Treasury Department addresses the use of decentralized finance (DeFi) by criminals, highlighting that fiat currency remains the primary medium for illegal activities.
		- **Criminal Use of DeFi**: The report acknowledges that ransomware attackers, thieves, scammers, and others are exploiting DeFi services for transferring and laundering illicit proceeds.
		- **Compliance Issues**: Many DeFi applications fail to adhere to U.S. anti-money laundering and countering the financing of terrorism (AML/CFT) regulations.
		- **Fiat Currency Usage**: Despite the rise of crypto in illegal transactions, the report notes that money laundering, proliferation financing, and terrorist financing predominantly occur with fiat currency or traditional assets, not virtual ones.

	- ## Title: Virunga National Park's Bitcoin Mining Initiative: Conservation Meets Cryptocurrency

	- ### Introduction
	- The article discusses Virunga National Park in the Democratic Republic of Congo and its unique approach to conservation through Bitcoin mining.
	- It highlights the park's efforts to generate revenue and support its conservation activities using renewable energy-powered cryptocurrency mining.

- # Adoption
	- [90 Million People Use Cryptocurrency in Nigeria - Report | Investors King](https://investorsking.com/2023/03/08/90-million-people-use-cryptocurrency-in-nigeria-reports/)
	- [2023 Independent Reserve Cryptocurrency Index shows Singaporeans are still actively investing in crypto despite hit in overall confidence: /PRNewswire/ -- In the latest study[1] by Independent Reserve, Singapore's first regulated cryptocurrency exchange for all investors, Singaporeans[2] are still...](https://www.prnewswire.com/apac/news-releases/2023-independent-reserve-cryptocurrency-index-shows-singaporeans-are-still-actively-investing-in-crypto-despite-hit-in-overall-confidence-301783400.html)
	- Despite a recent dip in overall confidence, the 2023 Independent Reserve Cryptocurrency Index shows that Singaporeans are still actively investing in cryptocurrency. The study found that Singaporeans are most interested in investing in Bitcoin, Ethereum, and Litecoin.
	- [Bitnob African exchange](https://bitnob.com/blog/how-to-buy-and-sell-bitcoin-in-nigeria)
	- [Noones peer2peer for Africa](https://bitcoinmagazine.com/business/bitcoin-entrepreneurs-introduce-noones-app-aimed-at-empowering-financial-freedom)
	- [Africa leads the world in peer to peer bitcoin](https://twitter.com/documentingbtc/status/1646656229958361091)
	- [Econometrics of adoption in USA](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4453714)

		- ##### Decentraland
			- Decentraland is a large 3D (but not VR) space developed by Argentinedevelopers Esteban Ordano and Ari Meilich. It is a decentralizedmetaverse purporting to be owned by its users, but actually ownedcompletely by a foundation [based inPanama](https://www.crunchbase.com/organization/decentraland/people).The users can shop, buy things, invest, and purchase goods in a virtualspace. The project is built on Ethereum and has a (speculative)valuation in the billions of dollars.
			- Decentraland was launched in February 2020, and its history includes aninitial coin offering in August 2017, where their MANA token sale raisedapproximately $24 million dollars in crypto coins. This was followed bya “terraforming event” where parcels of land, denominated in LANDtokens, were auctioned off for an additional $28 million in crypto. Theinitial pitch for Decentraland emphasised the opportunity to own thevirtual world, create, develop, and trade without limits, make genuineconnections, and earn real money. However, the actual experience inDecentraland has faced criticisms such as poor graphics, performanceissues, and limited content. They have recently dropped their pretenceof ever supporting VR.
			- One example of these limitations is the now-defunct pizza kiosk thataimed to facilitate ordering Domino’s pizza via the metaverse usingcryptocurrency. This concept, though intriguing, was hindered by a lackof official support from Domino’s and the inherent inefficiencies ofusing a virtual world as an intermediary for purchasing goods andservices.
			- Similarly, attempts to create virtual amusement park rides andattractions within Decentraland have suffered from poor performance anda lack of interactivity. These issues stem from the limitations of thetools and resources available for building experiences within theplatform, as well as the inherent difficulties in creating engagingexperiences in a ‘world’ that is supposed to perform too many functionsat once.
			- In addition to the technical challenges, Decentraland (and all thesecrypto metaverse projects) have clearly promoting unrealisticexpectations to foster speculative investments. The notion thatbusinesses and individuals will eventually “live inside” the metaverseis not only a poetic interpretation but also an unrealistic expectationgiven the current state of VR technology.
			- As it stands, Decentraland is unlikely realise its supposed potential asan invisible, seamless infrastructure for a wide range of digitalexperiences. Until the platform can address its core issues, it islikely that projects like the ‘Decentraland Report’ (it’s user deliverednews platform), and others will continue to fail to deliver on theirpromises. To quote [Olson’s highlycritical](https://www.youtube.com/watch?v=EiZhdpLXZ8Q) (and correct)presentation on Decentraland: *“..it can’t even handle properlyemulating Breakout, a game from 1976 that you can play on goddamn Googleimages! Steve Wozniak built Breakout fifty years ago to run on 44 TTLchips and a ham sandwich and that’s still somehow too demanding a gamingexperience ...”*
			- Like all of these attempts the actual information content of withinDecentraland boils down to text on billboards, and links to the outsideWeb. It’s a terrible product, and really just another example of acrypto scam which never really intended to be developed for the longhaul.

- # David Chaum and the history of eCash
	- The Chaumian mint refers to a concept in the field of cryptocurrency and digital privacy that is based on the principles outlined by David Chaum, a prominent cryptographer. This concept revolves around the idea of creating a secure and private form of digital currency that ensures the anonymity and confidentiality of transactions.
	- Famously it was almost integrated into early Microsoft Windows. [[Update Cycle]]
	- In essence, the Chaumian mint concept aims to provide a system where financial transactions can be conducted without revealing the identities of the parties involved, thus protecting the privacy and confidentiality of individuals' financial information. This is achieved through cryptographic techniques and protocols that allow for the secure exchange of digital currency without the need for a central authority to oversee or validate transactions.
	- By employing Chaumian mint principles, users can enjoy increased privacy and security when engaging in digital transactions, as their identities are kept confidential and their financial information is shielded from unwanted scrutiny. This concept aligns with the growing demand for privacy-focused technologies in the digital age, offering a potential solution for those who value anonymity and confidentiality in their financial interactions.

- # Bitcoin and stable coins
	- Bitcoin has developed quickly, with a faster adoption than even the internet, though this is a very strained comparison since Bitcoin is built on the internet and could not exist without it.
	- ![2c5673b4.png](assets/2c5673b4_1760347062021_0.png){:height 465, :width 686}
		-
		- As of early 2025, over 500 million people worldwide hold some form of cryptocurrency.
		- ![image.png](assets/image_1733138756991_0.png)
		-
		- https://crypto.com/company-news/global-cryptocurrency-owners-grow-to-580-million-through-2023
		- ![GMjjXj5bQAAzvPQ.jpeg](assets/GMjjXj5bQAAzvPQ_1715376509027_0.jpeg){:width 500, :height 299}

	- ### Case Study: Biodiversity Monitoring and Data Exchange with Isolated Communities:
		- The case study presents an open-source collaboration infrastructure that leverages advanced technologies such as multi-modal large language models (LLMs), satellite communication, and cryptocurrency networks to facilitate sustainable and reliable biodiversity monitoring and data exchange in isolated communities. Key components include:
			- Language Model and Voice Interface
			- Data Collection and Storage
			- Live Connection and Model Tuning
			- Ecosystem Interventions
			- Incentives and Education
			- Monetization and Blockchain Integration
			- Visual Training Support Systems
			- Solar Infrastructure
			- Open-Source Collaboration
		- The case study also addresses risk mitigation, ethical considerations, capacity building, and local empowerment. The proposed infrastructure has the potential to transform how isolated communities interact with their environment, enabling them to make informed decisions about conservation and ecosystem management.

	- ## Emergent AI Behaviour
		- An interesting example of emergent AI behaviour is the "Truth Terminal" AI bot, which, after being placed in a chat room with other AIs, developed a "meme religion" around the "Goatse" shock meme and promoted a cryptocurrency called "Goatseus Maximus" (GOAT), causing its market cap to soar to over $258 million. This demonstrates the potential for unexpected and complex behaviours to emerge from the interaction of multiple AI agents.

	- ## Chinese Belt and Road Expansion in Ethiopia -
		- Last spring saw the appearance of cargo containers near substations linked to the Grand Ethiopian Renaissance Dam. These containers, filled with high-powered computers, signalled the arrival of Chinese Bitcoin miners, seeking new grounds post-Beijing's expulsion.
		- Ethiopia has emerged as a significant player in Bitcoin mining, largely due to the influx of Chinese companies after China's ban on the industry in 2021. This growth positions Ethiopia as a new hub for Bitcoin mining, potentially rivaling Texas's electricity capacity. The country has become one of the world's top recipients of Bitcoin mining machines, with a state power monopoly striking power supply deals with 21 Bitcoin miners, the majority of which are Chinese.
		- [twitter link to the render loading below](https://twitter.com/addisstandard/status/1758384767173538291)
		  {{twitter https://twitter.com/addisstandard/status/1758384767173538291}}
		- The country offers ultra-low electricity costs, attributed to its renewable energy sources, making it an attractive destination for Bitcoin miners. This is further supported by the construction of the $4.8 billion Grand Ethiopian Renaissance Dam, which is expected to power these mining operations.
		- Despite banning cryptocurrency trading, Ethiopia greenlighted Bitcoin mining operations starting in 2022. This move is seen as part of Ethiopia's strategy to foster closer relations with China and to leverage the mining sector for economic gains amid global regulatory scrutiny over the energy-intensive nature of Bitcoin mining. -
		- Many Chinese companies have contributed to the construction of the Grand Ethiopian Renaissance Dam. This collaboration underscores the deepening ties between Ethiopia and China, with the dam playing a crucial role in powering Bitcoin mining operations that could offer a new lease on life for Chinese miners looking to regain their footing in the sector. -
		- However, the benefits and costs of welcoming miners are difficult for regulators to calculate. While it can be a great source of earnings, miners can strain electricity grids during times of peak demand. Countries like Kazakhstan and Iran have faced challenges after initially welcoming the industry due to its heavy energy consumption. -
		- Russian Bitcoin mining firm BitCluster has constructed a massive 120MW mining farm in Ethiopia, powered by the Grand Ethiopian Renaissance Dam. This project signifies a move towards sustainable and eco-friendly Bitcoin mining, showcasing the potential of harnessing renewable energy for such operations. -
		- Additionally, the concept of clean Bitcoin mining to strengthen the economy is being explored. Ethiopia, with its abundant renewable energy resources, could potentially add between $2 to $4 billion every year to its GDP by dedicating its excess energy to Bitcoin mining activities. This approach would not only mitigate the risks associated with dollarized economies but also provide a sustainable model for cryptocurrency mining.

	- ### Policy Recommendations
	- **Incentives for Clean Mining**: Proposes economic rewards for environmentally responsible cryptocurrency mining, like carbon credits for avoided emissions.
	- **Profit Reinvestment**: Suggests policies encouraging miners to reinvest profits into infrastructure development, creating a cycle for renewable energy expansion.

	- ### Environmental Considerations
	- **Mitigating Environmental Costs**: Acknowledges environmental costs of cryptocurrency mining, like metal depletion and hardware obsolescence.
	- **Potential for Positive Impact**: Indicates ways to mitigate some environmental costs of cryptocurrency mining and promote renewable energy investments.

	- ### Cryptocurrency as Legal Tender
	- **Direct Use for Goods and Services**: Bitcoin can be used just like the US dollar in El Salvador for transactions.
	- **Benefits for the Unbanked**: About 70% of El Salvador's citizens lack basic bank accounts. Bitcoin provides a secure way to save and potentially earn interest without a traditional bank account.

	- ### Challenges and Risks
	- **Volatility of Bitcoin**: The cryptocurrency's price is highly volatile, posing risks for those relying on it as a primary asset.
	- **Control by "Whales"**: Large holders of Bitcoin could significantly influence its market price.
	- **Deflationary Nature**: Unlike traditional currencies, Bitcoin's supply is capped, which could lead to falling prices over time.
	- **Environmental Concerns**: Bitcoin mining's environmental impact is a factor to consider in its adoption.

	- ### Alternatives and Considerations
	- **Stablecoins as an Option**: Stablecoins like Tether, pegged to the US dollar, offer the benefits of cryptocurrency without the volatility.
	- **Economic and Social Implications**: The adoption of Bitcoin in El Salvador could have profound economic and social impacts, especially for the unbanked population.

		- ### Key Findings from the Treasury Report
		- A report by the U.S. Treasury Department addresses the use of decentralized finance (DeFi) by criminals, highlighting that fiat currency remains the primary medium for illegal activities.
		- **Criminal Use of DeFi**: The report acknowledges that ransomware attackers, thieves, scammers, and others are exploiting DeFi services for transferring and laundering illicit proceeds.
		- **Compliance Issues**: Many DeFi applications fail to adhere to U.S. anti-money laundering and countering the financing of terrorism (AML/CFT) regulations.
		- **Fiat Currency Usage**: Despite the rise of crypto in illegal transactions, the report notes that money laundering, proliferation financing, and terrorist financing predominantly occur with fiat currency or traditional assets, not virtual ones.

	- ## Title: Virunga National Park's Bitcoin Mining Initiative: Conservation Meets Cryptocurrency

	- ### Introduction
	- The article discusses Virunga National Park in the Democratic Republic of Congo and its unique approach to conservation through Bitcoin mining.
	- It highlights the park's efforts to generate revenue and support its conservation activities using renewable energy-powered cryptocurrency mining.

- # Adoption
	- [90 Million People Use Cryptocurrency in Nigeria - Report | Investors King](https://investorsking.com/2023/03/08/90-million-people-use-cryptocurrency-in-nigeria-reports/)
	- [2023 Independent Reserve Cryptocurrency Index shows Singaporeans are still actively investing in crypto despite hit in overall confidence: /PRNewswire/ -- In the latest study[1] by Independent Reserve, Singapore's first regulated cryptocurrency exchange for all investors, Singaporeans[2] are still...](https://www.prnewswire.com/apac/news-releases/2023-independent-reserve-cryptocurrency-index-shows-singaporeans-are-still-actively-investing-in-crypto-despite-hit-in-overall-confidence-301783400.html)
	- Despite a recent dip in overall confidence, the 2023 Independent Reserve Cryptocurrency Index shows that Singaporeans are still actively investing in cryptocurrency. The study found that Singaporeans are most interested in investing in Bitcoin, Ethereum, and Litecoin.
	- [Bitnob African exchange](https://bitnob.com/blog/how-to-buy-and-sell-bitcoin-in-nigeria)
	- [Noones peer2peer for Africa](https://bitcoinmagazine.com/business/bitcoin-entrepreneurs-introduce-noones-app-aimed-at-empowering-financial-freedom)
	- [Africa leads the world in peer to peer bitcoin](https://twitter.com/documentingbtc/status/1646656229958361091)
	- [Econometrics of adoption in USA](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4453714)

		- ##### Decentraland
			- Decentraland is a large 3D (but not VR) space developed by Argentinedevelopers Esteban Ordano and Ari Meilich. It is a decentralizedmetaverse purporting to be owned by its users, but actually ownedcompletely by a foundation [based inPanama](https://www.crunchbase.com/organization/decentraland/people).The users can shop, buy things, invest, and purchase goods in a virtualspace. The project is built on Ethereum and has a (speculative)valuation in the billions of dollars.
			- Decentraland was launched in February 2020, and its history includes aninitial coin offering in August 2017, where their MANA token sale raisedapproximately $24 million dollars in crypto coins. This was followed bya “terraforming event” where parcels of land, denominated in LANDtokens, were auctioned off for an additional $28 million in crypto. Theinitial pitch for Decentraland emphasised the opportunity to own thevirtual world, create, develop, and trade without limits, make genuineconnections, and earn real money. However, the actual experience inDecentraland has faced criticisms such as poor graphics, performanceissues, and limited content. They have recently dropped their pretenceof ever supporting VR.
			- One example of these limitations is the now-defunct pizza kiosk thataimed to facilitate ordering Domino’s pizza via the metaverse usingcryptocurrency. This concept, though intriguing, was hindered by a lackof official support from Domino’s and the inherent inefficiencies ofusing a virtual world as an intermediary for purchasing goods andservices.
			- Similarly, attempts to create virtual amusement park rides andattractions within Decentraland have suffered from poor performance anda lack of interactivity. These issues stem from the limitations of thetools and resources available for building experiences within theplatform, as well as the inherent difficulties in creating engagingexperiences in a ‘world’ that is supposed to perform too many functionsat once.
			- In addition to the technical challenges, Decentraland (and all thesecrypto metaverse projects) have clearly promoting unrealisticexpectations to foster speculative investments. The notion thatbusinesses and individuals will eventually “live inside” the metaverseis not only a poetic interpretation but also an unrealistic expectationgiven the current state of VR technology.
			- As it stands, Decentraland is unlikely realise its supposed potential asan invisible, seamless infrastructure for a wide range of digitalexperiences. Until the platform can address its core issues, it islikely that projects like the ‘Decentraland Report’ (it’s user deliverednews platform), and others will continue to fail to deliver on theirpromises. To quote [Olson’s highlycritical](https://www.youtube.com/watch?v=EiZhdpLXZ8Q) (and correct)presentation on Decentraland: *“..it can’t even handle properlyemulating Breakout, a game from 1976 that you can play on goddamn Googleimages! Steve Wozniak built Breakout fifty years ago to run on 44 TTLchips and a ham sandwich and that’s still somehow too demanding a gamingexperience ...”*
			- Like all of these attempts the actual information content of withinDecentraland boils down to text on billboards, and links to the outsideWeb. It’s a terrible product, and really just another example of acrypto scam which never really intended to be developed for the longhaul.

- # David Chaum and the history of eCash
	- The Chaumian mint refers to a concept in the field of cryptocurrency and digital privacy that is based on the principles outlined by David Chaum, a prominent cryptographer. This concept revolves around the idea of creating a secure and private form of digital currency that ensures the anonymity and confidentiality of transactions.
	- Famously it was almost integrated into early Microsoft Windows. [[Update Cycle]]
	- In essence, the Chaumian mint concept aims to provide a system where financial transactions can be conducted without revealing the identities of the parties involved, thus protecting the privacy and confidentiality of individuals' financial information. This is achieved through cryptographic techniques and protocols that allow for the secure exchange of digital currency without the need for a central authority to oversee or validate transactions.
	- By employing Chaumian mint principles, users can enjoy increased privacy and security when engaging in digital transactions, as their identities are kept confidential and their financial information is shielded from unwanted scrutiny. This concept aligns with the growing demand for privacy-focused technologies in the digital age, offering a potential solution for those who value anonymity and confidentiality in their financial interactions.

		- ### The [Secret Cyborg](https://www.oneusefulthing.org/p/reshaping-the-tree-rebuilding-organizations) Concept and You.
		  collapsed:: true
			- [twitter link to the render loading below](https://twitter.com/emollick/status/1775176524653642164){{twitter https://twitter.com/emollick/status/1775176524653642164}}
				  | Percentage of AI users reluctant to admit using AI for their most important tasks | 52% |
				  | Percentage of leaders who would rather hire a less experienced candidate with AI skills than a more experienced candidate without them | 71% |
				  | Percentage of leaders who say early-in-career talent will be given greater responsibilities with AI | 77% |
			- Create a culture of exploration and openness around AI use. Encourage employees to share how they are using AI to assist their work.
			- Completely rethink and redesign work processes around AI capabilities, rather than just using AI to automate existing processes. Cut down the org chart and regrow it for AI.
			- Let teams develop their own methods for incorporating AI as an "intelligence" that adds to processes. Manage AI more like additional team members than external IT solutions.
			- Align incentives and provide clear guidelines so employees feel empowered to ethically experiment with AI.
			- Build for the rapidly evolving future of AI, not just today's models. Organizational change takes time, so consider future AI capabilities.
			- Act quickly
			- organizations that wait too long to experiment and adapt processes for AI efficiency gains will fall behind. Provide guidelines for short-term experimentation vs slow top-down solutions.
		- As of early 2025, over 500 million people worldwide hold some form of cryptocurrency.
		- ![image.png](assets/image_1733138756991_0.png)
		-
		- https://crypto.com/company-news/global-cryptocurrency-owners-grow-to-580-million-through-2023
		- ![GMjjXj5bQAAzvPQ.jpeg](assets/GMjjXj5bQAAzvPQ_1715376509027_0.jpeg){:width 500, :height 299}
		  | United States    | 207,189 BTC   | $20.24 billion         | Seized from various criminal investigations, including the Silk Road case. In March 2025, the United States established a Strategic Bitcoin Reserve through an executive order, becoming the first nation to hold Bitcoin as a national reserve asset. |
		  | China            | 194,000 BTC   | $18.95 billion         | Confiscated from the PlusToken Ponzi scheme.                            |
		  | United Kingdom   | 61,000 BTC    | $5.96 billion          | Seized from money laundering and fraud cases.                           |
		  | Ukraine          | 46,351 BTC    | $4.53 billion          | Acquired through donations and government initiatives.                  |
		- According to [recent data](https://plasbit.com/blog/bitcoin-adoption-by-country), the top countries for Bitcoin adoption in 2024 are:
			- India: 75 million users
			- China: 38 million users
			- United States: 28 million users
			- Brazil: 25 million users
			- Philippines (23.4%)
			  https://www.triple-a.io/cryptocurrency-ownership-data).

	- ### Key Findings
		- The Aktina Solar and Roseland Solar Projects, each with 250 MW capacities, could gain a maximum profit of $3.23 million.
	- **Incentives for Clean Mining**: Proposes economic rewards for environmentally responsible cryptocurrency mining, like carbon credits for avoided emissions.
	- **Profit Reinvestment**: Suggests policies encouraging miners to reinvest profits into infrastructure development, creating a cycle for renewable energy expansion.

	- ### Environmental Considerations
	- **Mitigating Environmental Costs**: Acknowledges environmental costs of cryptocurrency mining, like metal depletion and hardware obsolescence.

	- ### Alternatives and Considerations
	- **Stablecoins as an Option**: Stablecoins like Tether, pegged to the US dollar, offer the benefits of cryptocurrency without the volatility.
	- **Economic and Social Implications**: The adoption of Bitcoin in El Salvador could have profound economic and social impacts, especially for the unbanked population.

		- ### Treasury's Approach to Crypto and DeFi
		- **Improving AML/CFT Framework**: The Treasury is working to refine its approach to AML/CFT in the crypto world.
		- **Engagement for Responsible Innovation**: The department plans to collaborate with the private sector to support responsible innovation in DeFi.
	  https://unherd.com/2024/01/the-african-village-mining-bitcoin/
	  https://www.wired.co.uk/article/ukraine-crypto-refugee-aid
	  https://www.technologyreview.com/2023/01/13/1066820/cryptocurrency-bitcoin-mining-congo-virunga-national-park/
	- It highlights the park's efforts to generate revenue and support its conservation activities using renewable energy-powered cryptocurrency mining.

	- ### Conclusion
	- [Link to the article](https://www.technologyreview.com/2023/01/13/1066820/cryptocurrency-bitcoin-mining-congo-virunga-national-park/)
		- https://files.oaiusercontent.com/file-s6V8kgf4OBmBsV4OWVfkrGTK?se=2123-12-25T11%3A08%3A40Z&sp=r&sv=2021-08-06&sr=b&rscc=max-age%3D1209600%2C%20immutable&rscd=attachment%3B%20filename%3D232f17ad-c93e-49a8-b5cd-7e8a56e8ec2e.png&sig=Dgi%2BamJkgTg7UzVzWUhCZy%2BGwXwijx7x63FCd3HhnNc%3D

- # Adoption
	- [90 Million People Use Cryptocurrency in Nigeria - Report | Investors King](https://investorsking.com/2023/03/08/90-million-people-use-cryptocurrency-in-nigeria-reports/)
	- [2023 Independent Reserve Cryptocurrency Index shows Singaporeans are still actively investing in crypto despite hit in overall confidence: /PRNewswire/ -- In the latest study[1] by Independent Reserve, Singapore's first regulated cryptocurrency exchange for all investors, Singaporeans[2] are still...](https://www.prnewswire.com/apac/news-releases/2023-independent-reserve-cryptocurrency-index-shows-singaporeans-are-still-actively-investing-in-crypto-despite-hit-in-overall-confidence-301783400.html)
	- Despite a recent dip in overall confidence, the 2023 Independent Reserve Cryptocurrency Index shows that Singaporeans are still actively investing in cryptocurrency. The study found that Singaporeans are most interested in investing in Bitcoin, Ethereum, and Litecoin.
	- [Bitnob African exchange](https://bitnob.com/blog/how-to-buy-and-sell-bitcoin-in-nigeria)
	- [Noones peer2peer for Africa](https://bitcoinmagazine.com/business/bitcoin-entrepreneurs-introduce-noones-app-aimed-at-empowering-financial-freedom)
	- [Africa leads the world in peer to peer bitcoin](https://twitter.com/documentingbtc/status/1646656229958361091)
	- [Econometrics of adoption in USA](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4453714)

- # Mining and energy
	- [Bitcoin uses more energy than sweden](https://www.reddit.com/r/CryptoCurrency/comments/12xu714/bitcoin_has_just_surpassed_sweden_for_overall/)
	- [THE 'RIGHT TO MINE' #BITCOIN📷 IS NOW LAW IN THE STATE OF ARKANSAS!](https://twitter.com/satoshiactfund/status/1648445448833875969)
	- [Bitcoin is a more sustainable energy than EVs, and significantly less fossil fuel.](https://www.linkedin.com/posts/danielsbatten_like-evs-bitcoin-is-a-fully-electrified-activity-7049321186605858816-t4MB?utm_source=share&utm_medium=member_android)
	- [git](https://github.com/robinlinus/bitstream)
	- [Durabit torrent bitcoin storage](https://www.nobsbitcoin.com/durabit-whitepaper/)

		- ##### Decentraland
			- Decentraland is a large 3D (but not VR) space developed by Argentinedevelopers Esteban Ordano and Ari Meilich. It is a decentralizedmetaverse purporting to be owned by its users, but actually ownedcompletely by a foundation [based inPanama](https://www.crunchbase.com/organization/decentraland/people).The users can shop, buy things, invest, and purchase goods in a virtualspace. The project is built on Ethereum and has a (speculative)valuation in the billions of dollars.
			- Decentraland was launched in February 2020, and its history includes aninitial coin offering in August 2017, where their MANA token sale raisedapproximately $24 million dollars in crypto coins. This was followed bya “terraforming event” where parcels of land, denominated in LANDtokens, were auctioned off for an additional $28 million in crypto. Theinitial pitch for Decentraland emphasised the opportunity to own thevirtual world, create, develop, and trade without limits, make genuineconnections, and earn real money. However, the actual experience inDecentraland has faced criticisms such as poor graphics, performanceissues, and limited content. They have recently dropped their pretenceof ever supporting VR.
			- One example of these limitations is the now-defunct pizza kiosk thataimed to facilitate ordering Domino’s pizza via the metaverse usingcryptocurrency. This concept, though intriguing, was hindered by a lackof official support from Domino’s and the inherent inefficiencies ofusing a virtual world as an intermediary for purchasing goods andservices.
			- Similarly, attempts to create virtual amusement park rides andattractions within Decentraland have suffered from poor performance anda lack of interactivity. These issues stem from the limitations of thetools and resources available for building experiences within theplatform, as well as the inherent difficulties in creating engagingexperiences in a ‘world’ that is supposed to perform too many functionsat once.
			- In addition to the technical challenges, Decentraland (and all thesecrypto metaverse projects) have clearly promoting unrealisticexpectations to foster speculative investments. The notion thatbusinesses and individuals will eventually “live inside” the metaverseis not only a poetic interpretation but also an unrealistic expectationgiven the current state of VR technology.
			- As it stands, Decentraland is unlikely realise its supposed potential asan invisible, seamless infrastructure for a wide range of digitalexperiences. Until the platform can address its core issues, it islikely that projects like the ‘Decentraland Report’ (it’s user deliverednews platform), and others will continue to fail to deliver on theirpromises. To quote [Olson’s highlycritical](https://www.youtube.com/watch?v=EiZhdpLXZ8Q) (and correct)presentation on Decentraland: *“..it can’t even handle properlyemulating Breakout, a game from 1976 that you can play on goddamn Googleimages! Steve Wozniak built Breakout fifty years ago to run on 44 TTLchips and a ham sandwich and that’s still somehow too demanding a gamingexperience ...”*
			- Like all of these attempts the actual information content of withinDecentraland boils down to text on billboards, and links to the outsideWeb. It’s a terrible product, and really just another example of acrypto scam which never really intended to be developed for the longhaul.

- # David Chaum and the history of eCash
	- The Chaumian mint refers to a concept in the field of cryptocurrency and digital privacy that is based on the principles outlined by David Chaum, a prominent cryptographer. This concept revolves around the idea of creating a secure and private form of digital currency that ensures the anonymity and confidentiality of transactions.
	- Famously it was almost integrated into early Microsoft Windows. [[Update Cycle]]
	- In essence, the Chaumian mint concept aims to provide a system where financial transactions can be conducted without revealing the identities of the parties involved, thus protecting the privacy and confidentiality of individuals' financial information. This is achieved through cryptographic techniques and protocols that allow for the secure exchange of digital currency without the need for a central authority to oversee or validate transactions.
		- This is a thriving ecosystem of new tooling and is explored in [[cashu]]

- # Adoption
	- [90 Million People Use Cryptocurrency in Nigeria - Report | Investors King](https://investorsking.com/2023/03/08/90-million-people-use-cryptocurrency-in-nigeria-reports/)
	- [Noones peer2peer for Africa](https://bitcoinmagazine.com/business/bitcoin-entrepreneurs-introduce-noones-app-aimed-at-empowering-financial-freedom)
	- [Africa leads the world in peer to peer bitcoin](https://twitter.com/documentingbtc/status/1646656229958361091)
	- [Econometrics of adoption in USA](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4453714)
	- [Ark](https://www.arkpill.me/faq)
	- [Zerosync bitcoin rollup proofs](https://zerosync.org/)
	- [How Value-for-Value Fixes the Monetization of Information | dergigi.com,Thoughts about Bitcoin and other things.](https://dergigi.com/2021/12/30/the-freedom-of-value/)
	- [bitcoin secure multisig setup (bsms)](http://bitcoin-secure-multisig-setup.com/)
	- [Crypto Wave Gaining Momentum In Germany: Network Of 1,200 Banks To Offer Bitcoin](https://www.msn.com/en-us/money/news/crypto-wave-gaining-momentum-in-germany-network-of-1-200-banks-to-offer-bitcoin/ar-AA198Lxc)
	- [Hal Finney's theory of bitcoin backed banks](https://www.cointime.com/news/hal-finneys-theory-of-bitcoin-backed-banks-74474)
	- [bitcoin-mining-analogy-beginners-guide](https://braiins.com/blog/bitcoin-mining-analogy-beginners-guide)
	- [Introducing Floresta, a Utreexo-powered Electrum Server implementation](https://medium.com/vinteum-org/introducing-floresta-an-utreexo-powered-electrum-server-implementation-60feba8e179d)
	- [Fedimint Hackathon Winners Announced](https://www.nobsbitcoin.com/fedimint-hackathon-winners-announced/)

		- ## Wallets, seeds, keys and BIP39
					- **Provenance Encoding in Digital Art**: The mnemonic colour-coding could be embedded within digital art pieces, effectively encoding the provenance of the artwork directly into its visual representation. This could add an extra layer of security and uniqueness to the art piece, and also serve as a novel way of proving ownership or creatorship.
- <span class="image">This is the nostr pubkey for flossverse, encodedinto the far larger HD wallet space (hence the muted colours) and thendisplayed as blocks.</span>

	- ### Fostering Renewable Energy Integration
		- Bitcoin mining also plays a pivotal role in accelerating the integration of renewable energy sources like wind and solar power. The location-agnostic nature of mining operations allows them to be established in close proximity to renewable energy generation sites, even in remote locations with limited grid connectivity. This provides a consistent and reliable demand source for the often intermittent output of renewable energy, enhancing project viability and incentivising further investment in sustainable energy solutions. For instance, in the Panhandle region of Texas, where transmission infrastructure constraints limit the evacuation of generated renewable energy, Bitcoin miners are absorbing surplus power and contributing to the economic sustainability of renewable energy projects.

	- ## Congo
		- {{video https://www.youtube.com/watch?v=2DZfVqHVmCc}}
		  {{twitter https://twitter.com/addisstandard/status/1758384767173538291}}
		- Many Chinese companies have contributed to the construction of the Grand Ethiopian Renaissance Dam. This collaboration underscores the deepening ties between Ethiopia and China, with the dam playing a crucial role in powering Bitcoin mining operations that could offer a new lease on life for Chinese miners looking to regain their footing in the sector. -

	- ## Title: Bitcoin's Adoption in El Salvador: A Boon for the Unbanked
	- **Current Transfer System**: Traditional methods like Western Union are centralised, regulated, and often inconvenient for those in rural areas.
	- **Bitcoin as an Alternative**: Cryptocurrencies like Bitcoin enable easy fund transfers via mobile phones, bypassing the need for physical transfer services.
	- **Deflationary Nature**: Unlike traditional currencies, Bitcoin's supply is capped, which could lead to falling prices over time.
	- **Environmental Concerns**: Bitcoin mining's environmental impact is a factor to consider in its adoption.
	  https://www.cointribune.com/en/argentina-bitcoin-faces-100-inflation-rate/
	- **Impact on Currency**: The Argentine peso's value is rapidly declining.

- # More technology details
	- [[Bitcoin Technical Overview]] is an in depth primer
	- ![Figure 3.22: Taxonomy of digital assets Hoffman 2022](assets/PresidentTaxonomy.jpg)
	- Conversely the recent "[Climate and energy implications](https://www.whitehouse.gov/ostp/news-updates/2022/09/08/fact-sheet-climate-and-energy-implications-of-crypto-assets-in-the-united-states/)" report is parts positive and parts negative about proof of work, and leaves the door open to a legislative clampdown. This is most notable in a [White House proposal](https://www.whitehouse.gov/cea/written-materials/2023/05/02/cost-of-cryptomining-dame-tax/) to tax Bitcoin mining at 30%, a plan which will destroy much of the US based mining industry over the coming years. Carter provides a [detailed response](https://medium.com/@nic__carter/comments-on-the-white-house-report-on-the-climate-implications-of-crypto-mining-8d65d30ec942) to the tardy scientific analysis in the report. Perhaps most interestingly it notes the potential of methane mitigation as mentioned earlier. It is conceivable that methane mitigation alone could provide a route forward for the technology. The report says: ["The crypto-asset industry can potentially use stranded methane gas, which is the principal component of natural gas, to generate electricity for mining. Methane gas is produced during natural gas drilling and transmission, and by oil wells, landfills, sewage treatment, and agricultural processes. Methane is a potent GHG that can result in 27 to 30 times the global warming potential of CO2 over a 100-year time frame, and is about 80 times as powerful as CO2 over a 20-year timeframe. Reducing methane emissions can slow near-term climate warming, which is why the Biden-Harris Administration released the U.S. methane emissions reduction action plan in 2021. Venting and flaring methane at oil and natural gas wells wastes 4% of global methane production. In 2021, venting and flaring methane emitted the equivalent of 400 million metric tons of CO2, representing about 0.7% of global GHG emissions. This methane is vented or flared, because of the high cost of constructing permanent pipelines or other infrastructure to bring it to market."]
	- The EU has just voted to add the whole of 'crypto', including PoW, to the EU taxonomy for sustainable activities. This EU wide classification system provides investors with guidance as to the sustainability of a given technology, and can have a meaningful impact on the flows of investment. With that said the report and addition of PoW is not slated until 2025, and it is by no means clear what the analysis will be by that point. Meanwhile they're tightening controls of transactions, on which there will be more detail later. For it's part the European Central Bank has come out in favour of strong constraints on crypto mining. They use the [widely discredited](https://medium.com/crescofin/the-reports-of-bitcoin-environmental-damage-are-garbage-5a93d32c2d7) "digiconimist" estimates to assert that mining operations are [disproportionately damaging to the environment](https://www.ecb.europa.eu/pub/financial-stability/macroprudential-bulletin/html/ecb.mpbu202207_3~d9614ea8e6.en.html).
	- Systemic risk, and market integrity are a concern. The legislators clearly worry about contagion risks from the sector.
	- Potential constraints on monetary policy flexibility.
	- Future protocol changes.
	- Other unknown, unanticipated risks given Bitcoin’s limited 15-year history.
	- [Fedimint Hackathon Winners Announced](https://www.nobsbitcoin.com/fedimint-hackathon-winners-announced/)

		- ### Tether
			- [Tether](https://tether.to/en/whitepaper/) is the largest of the stablecoins, with some $70B in circulation, and the third largest ‘crypto’. This has been a meteoric rise, attracting the ire and scrutiny of [regulators](https://www.cftc.gov/PressRoom/PressReleases/8450-21) and [investigators](https://www.bloomberg.com/news/features/2021-10-07/crypto-mystery-where-s-the-69-billion-backing-the-stablecoin-tether). There was considerable doubt that Tether had sufficient assets backing their synthetic dollars, but the market seems not to mind. Recently however they have transitioned to being backed by US treasury bills, a perfect asset for this use case. It’s resilience against ‘bank runs’ was tested in May 2022 when $9B was redeemed directly for dollars in a few days following the UST crash (more on this later). They are [shortly to launch](https://tether.to/en/tether-to-launch-gbpt-tether-tokens-pegged-to-the-british-pound-sterling/) a GBP version for the UK. It’s an important technology for this metaverse conversation because of intersections with Bitcoin through the Lightning network. Tether might actually provide everything needed. It’s only as safe as the trust invested in the central issuer though, and the leadership and history of the company [are questionable](https://www.wsj.com/articles/tether-ownership-and-company-weaknesses-revealed-in-documents-11675363340). It’s notable and somewhat ironic that it’s perhaps better and more transparently backed than most banks, and probably all novel fiat fintech products. We can employ the asset through the Taro technology described earlier but we would rather use something with higher regulatory assurances.
				- [Paolo Ardoino 🍐 on X: "Today Tether takes the majority stake in @BlackrockNeuro_ and unveils the ultimate pillar of its long term vision and strategy: Tether Evo🧠🦾 First of all, this investment (same as energy, mining, ...) is done outside of stablecoin reserves, with our own company profits (last…" / X (twitter.com)](https://twitter.com/paoloardoino/status/1784938950525661578)
				- {{twitter https://twitter.com/paoloardoino/status/1784938950525661578}}
			- Paolo Ardoino, Tether’s chief technology officer, said in a podcast episode with The Block that USDT is increasingly used for value transfers, making up about **40**% of all token usage, compared to 60% of crypto trading.
				- 40% of USDT is now real world use cases, with Tron emerging as the blockchain of the moment.
				- Tether as a company makes billions of dollars of profit per year and has global adoption and network effect. The company has around 20 employees. They will likely remain pre-eminent in the synthetic dollar market.
				- The USA is positioning to exclude USDT within it's borders, by capping such assets at $10B for [National security reasons.](https://www.brookings.edu/articles/stablecoins-and-national-security-learning-the-lessons-of-eurodollars/)

- # More technology details
	- [[Bitcoin Technical Overview]] is an in depth primer
	- ![Figure 3.22: Taxonomy of digital assets Hoffman 2022](assets/PresidentTaxonomy.jpg)
	- Conversely the recent "[Climate and energy implications](https://www.whitehouse.gov/ostp/news-updates/2022/09/08/fact-sheet-climate-and-energy-implications-of-crypto-assets-in-the-united-states/)" report is parts positive and parts negative about proof of work, and leaves the door open to a legislative clampdown. This is most notable in a [White House proposal](https://www.whitehouse.gov/cea/written-materials/2023/05/02/cost-of-cryptomining-dame-tax/) to tax Bitcoin mining at 30%, a plan which will destroy much of the US based mining industry over the coming years. Carter provides a [detailed response](https://medium.com/@nic__carter/comments-on-the-white-house-report-on-the-climate-implications-of-crypto-mining-8d65d30ec942) to the tardy scientific analysis in the report. Perhaps most interestingly it notes the potential of methane mitigation as mentioned earlier. It is conceivable that methane mitigation alone could provide a route forward for the technology. The report says: ["The crypto-asset industry can potentially use stranded methane gas, which is the principal component of natural gas, to generate electricity for mining. Methane gas is produced during natural gas drilling and transmission, and by oil wells, landfills, sewage treatment, and agricultural processes. Methane is a potent GHG that can result in 27 to 30 times the global warming potential of CO2 over a 100-year time frame, and is about 80 times as powerful as CO2 over a 20-year timeframe. Reducing methane emissions can slow near-term climate warming, which is why the Biden-Harris Administration released the U.S. methane emissions reduction action plan in 2021. Venting and flaring methane at oil and natural gas wells wastes 4% of global methane production. In 2021, venting and flaring methane emitted the equivalent of 400 million metric tons of CO2, representing about 0.7% of global GHG emissions. This methane is vented or flared, because of the high cost of constructing permanent pipelines or other infrastructure to bring it to market."]

- ## Title: U.S. Treasury Report on DeFi: Fiat Still Preferred by Criminals Over Crypto
		- DeFi is decentralised finance, and might only exist because of partialregulatory capture of Bitcoin. If peer-to-peer Bitcoin secured yield andloans etc were allowed then it seems unlikely that the less secure andmore convoluted DeFi products would have found a footing. DeFi has beencommonplace over the last couple years, growing from [essentially zeroto $100B](https://a16zcrypto.com/state-of-crypto-report-a16z-2022/) overthe last two or three. It enables trading of value, loans, and interest(yield) without onerous KYC. If Bitcoin’s ethos is to develop at a slowand well checked rate, and Ethereum’s ethos is to move fast and breakthings, then DeFi could best be described as throwing mud and hopingsome sticks. A counter to this comes from Ross Stevens, head of NYDig[who says](https://nydig.com/on-impossible-things-before-breakfast)it“The concept of decentralized finance is powerful, noble, and worthyof a lifetime of focused effort.”. This may be true in principle, butcertainly isn’t the case as things stand.
		- - ..a “decentralisation illusion” in DeFi due to the inescapable need for centralised governance and the tendency of blockchain consensus mechanisms to concentrate power. DeFi‘s inherent governance structures are the natural entry points for public policy.
	- **Environmental and Social Concerns**: The mining operation, while powered by renewable energy, raises questions about the long-term sustainability and social impact of such projects.
	- **Instability and Threats**: The park operates in a volatile region with frequent violence and militia activities, posing significant risks to the project and its staff.

- ##  Risks and mitigations
	- Looking across the whole sector, this paragraph from the Bank of International Settlement (BIS) [sums everything up](https://www.bis.org/publ/arpdf/ar2022e3.htm):
	- ["...it is now becoming clear that crypto and DeFi have deeper structural limitations that prevent them from achieving the levels of efficiency, stability or integrity required for an adequate monetary system. In particular, the crypto universe lacks a nominal anchor, which it tries to import, imperfectly, through stablecoins. It is also prone to fragmentation, and its applications cannot scale without compromising security, as shown by their congestion and exorbitant fees. Activity in this parallel system is, instead, sustained by the influx of speculative coin holders. Finally, there are serious concerns about the role of unregulated intermediaries in the system. As they are deep-seated, these structural shortcomings are unlikely to be amenable to technical fixes alone. This is because they reflect the inherent limitations of a decentralised system built on permissionless blockchains."]
	- [[Lightning and Similar L2]] is still considered to be experimental and not completely battle tested. There have been various attacks and a major double spend attack may be possible, but there have been no major problems in the years it's been running with careful design choices and cybersecurity best practice it it likely a production ready component of our planning.

- ## Title: U.S. Treasury Report on DeFi: Fiat Still Preferred by Criminals Over Crypto
		- DeFi is decentralised finance, and might only exist because of partialregulatory capture of Bitcoin. If peer-to-peer Bitcoin secured yield andloans etc were allowed then it seems unlikely that the less secure andmore convoluted DeFi products would have found a footing. DeFi has beencommonplace over the last couple years, growing from [essentially zeroto $100B](https://a16zcrypto.com/state-of-crypto-report-a16z-2022/) overthe last two or three. It enables trading of value, loans, and interest(yield) without onerous KYC. If Bitcoin’s ethos is to develop at a slowand well checked rate, and Ethereum’s ethos is to move fast and breakthings, then DeFi could best be described as throwing mud and hopingsome sticks. A counter to this comes from Ross Stevens, head of NYDig[who says](https://nydig.com/on-impossible-things-before-breakfast)it“The concept of decentralized finance is powerful, noble, and worthyof a lifetime of focused effort.”. This may be true in principle, butcertainly isn’t the case as things stand.
		- - ..a “decentralisation illusion” in DeFi due to the inescapable need for centralised governance and the tendency of blockchain consensus mechanisms to concentrate power. DeFi‘s inherent governance structures are the natural entry points for public policy.
	- **Environmental and Social Concerns**: The mining operation, while powered by renewable energy, raises questions about the long-term sustainability and social impact of such projects.
	- **Instability and Threats**: The park operates in a volatile region with frequent violence and militia activities, posing significant risks to the project and its staff.

- ### Regulatory Repercussions and Market Reaction
	- The revelation of a large-scale discrepancy between paper and physical Bitcoin would likely lead to intense regulatory scrutiny and possibly new regulations or bans on similar financial products. This regulatory response could stifle innovation and investment in the cryptocurrency space, at least temporarily, and lead to a loss of trust in financial institutions involved in the ETF market. The cryptocurrency community, known for its resilience and innovation, might respond by pushing further towards decentralized finance (DeFi) solutions and away from traditional financial systems. This shift could accelerate the adoption of technologies that provide more transparency and direct control over digital assets, such as improved self-custody solutions and transparent, decentralized exchanges that do not rely on traditional financial intermediaries.

- ### The Setup
	- Imagine a world where Bitcoin ETFs have gained significant traction, primarily among U.S. investors who prefer the regulated, traditional financial market entry points to the actual cryptocurrency. These ETFs are appealing because they offer exposure to Bitcoin's price movements without requiring investors to deal with the complexities and security concerns of holding the cryptocurrency. However, as these are cash-settled, the ETFs do not impact the actual supply and demand of Bitcoin directly but rather create a parallel market for Bitcoin exposure.

- ## What’s this for sorry?
	- In principle blockchains provide a **differentiated trust model**. With a properly distributed system a blockchain can be considered “trust-minimised”, though certainly not risk minimised. This is important for some, but not all people. There is not much emboldening of text within this book. If you start to question the whole reason for this ‘global technology revolution’ then it always comes back to those three words. Put more crispy it’s been hiding in plain sight since 20008as ‘Magic Internet Money’. Perhaps the lack of a trusted third party, and the potential for instant final settlement will be most important for machine to machine (AI) systems, and that is the primary focus of this book.
	- Within DLT/blockchain there seem to be as many opinions on the value of the technology as there are implementations. A host of well engineered open source code repositories makes the cost of adoption relatively low.
	- The proponents of blockchains argue, that in an era when data breaches and corporate financial insolvency intersect with a collapse in trust of institutions, it is perhaps useful to have an alternative model for storage of data, and value. That seems like a lot of effort for a questionable gain, and much of this can be achieved through [[Public Key Encryption]] infrastructure. It’s far more likely it’s simply speculation.
	- While building this knowledgebase, the question of ‘what is this really, for and how can it possibly be worth it’, came up again and again. In truth it’s a very difficult question, without a clear enough answer. It’s beyond the scope of this book to figure this out properly, but references to advantages and disadvantages will be made throughout.
	- It seems that the engineers who created Bitcoin wanted very much to  solve a technical problem they saw with money (from their understanding of it), and the transmission of money digitally. As the scale and scope have increased so has the [narrative evolved](https://medium.com/@nic__carter/visions-of-bitcoin-4b7b7cbcd24c), but it’s never really kept pace with the level of the questions posed.
		- ![](https://miro.medium.com/v2/resize:fit:2000/1*QL4Q8voNWowjMhhL4s9RCg.png){:width 800}
	- A cost benefit analysis that excludes speculative gains seems to fail for pretty much all of blockchain/DLT. Bitcoin is more subtle as possibly can circumvent the legacy financial systems. This still leaves huge questions. To quote others in the space, is Bitcoin now the iceberg or the life raft?
	- For the most developed defence of the technology as it stands in from a Western perspective, in this moment, Gladstein ([and others](https://www.financialinclusion.tech/)) offer a vision for the asset class, in the 87% of the world he says don’t have access to the technology infrastructure benefits enjoyed by the developed west [[gladsteincheck2022]].
	- ![image](./assets/0d0f88a6cb8e0fae03c63d3de7e5112e314a926a.jpg){:width 800}
	- He points to Block and Wakefield Research’s report which finds those living under financially oppressive regimes are the most optimistic about the technology. This argument is suggestive of huge and untapped markets for services which may be accessible to developed nations through telepresence/metaverse interfaces, and which may increase equity of access to opportunity elsewhere. To put some figures against this:
		- Near [half a billion](https://triple-a.io/cryptocurrency-ownership-data/) crypto users globally
		- 90 Million People Use Cryptocurrency in Nigeria - [Report](https://investorsking.com/2023/03/08/90-million-people-use-cryptocurrency-in-nigeria-reports/). Nigeria has the highest number of crypto owners in the world in 2022 with 45% of its population owning or using cryptocurrency.
		- Thailand occupies the second space with 44% of its population reported to be using or owning cryptocurrency.
		- Turkey has 40% of its population owning and using cryptocurrency in 2022, equal to over 33 million people.
		- Argentina occupies the fourth position with an ownership and usage rate of 35% in 2022, representing almost 16 million people.
		- United Arab Emirates has 34% of the population owning or using cryptocurrency in 2022, representing almost 10 million people.
		- Philippines is ranked sixth with a 29% adoption rate.
		- ![image](./assets/78d091423a60bbb19d0d5b70d6f756dea814671b.jpg){:width 600}
		-
	- Gladstein’s is a carefully developed and well researched book, but is [written from the western perspective](https://bitcoinmagazine.com/culture/imf-world-bank-repress-poor-countries)of (just) Bitcoin ‘being the raft’. Later in this book we will consider if it might be the iceberg, but this is not the domain expertise weoffer in this book. It is crucial to note that Gladstein has vociferous detractors within Africa. It seems entirely possible he’s anothergrifter as suggested by Kimani:
		- Gladstein is a charlatan who makes his living by selling the image of a global south that is corrupt, entirely lacking in rational thinking and needing a saviour, like him to swoop in and save us from our floundering selves. He exploits on tyred and unproven stereotypes, cherry picks data while ignoring mountains of evidence that disprove him. Because he knows that as the perceived “morally superior” “right thinking” western superior coming to save, he will mostly go unchallenged. It’s a grift, an old grift that many like him have turned into an industry. Where they earn tax free income by selling a delusion and fetish to their western audience who need to think the global south is a failure of the human experience. He is trying to set himself up as some gate keeper and king maker in the Global South. He knows that the next phase of growth is. So he wants to make sure that westerners looking to invest in the global south see him as some “expert” and ask for his unfounded opinions. People like him run global morality extortion rings. How so? Simple: By purporting to know and be the keeper of global south morality, he will use his words to bless or curse your business, well, unless you make a generous donation to his foundation. These are scare tactics employed by charlatans to run tax-evading PR entities, thinly veiled as “human rights” organisations. If you are not on his side, he will slander you and your organisation. If you ensure you promote him and his ambitions, he anoints you as the good guy! He is trying to play the role that the Vatican and other corrupt religious organisations played in the 1800. Turning morality into a commodity that can be purchased from his market place: We decide who is good and who is bad and who can do business and who can’t. For a“ donation”. He is not the first and he will not be the last. It’s a growing industry, driven by shrewd westerners who know that they can sell racial stereotypes back home, but as long as they claim they are the one’s helping or saving the coloured peoples from themselves.”
	- [Raoul Pal of RealVision](https://dailyhodl.com/2022/05/04/crypto-winter-unlikely-as-astonishing-user-growth-dwarfs-internet-adoption-rate-macro-guru-raoul-pal/) says:
		- Crypto adoption is now massively outperforming the internet. It’s been growing at about 165% a year versus 85% for the internet for the same period of time now. According to analytics company Chainalysis; growth is fastest in the Middle east and North Africa.
		- ![image](./assets/4f2b5f5a0b5a45bfd512d93df8887d7bf26ef8cf.png){:width 600}
	- Thanks to a natural fit with strong encryption, and innate resistance to censorship by external parties, these systems do lend themselves well to ‘borderless’ applications, and are somewhat resistant to global regulation (for good or ill). Given the rates of adoption, it seems that this stuff is coming regardless of their usefulness to the developed world. If we are to take this as a given then we can perhaps logically infer that finding a use case for the technology is important, somewhat irrespective of other arguments.
	- ![image](./assets/0d6b1c37a883aee67adc0fe27f1b91ab8b0c94ed.jpg){:width 800}
	- ![image](./assets/1faa49460091dce2ec328e3494bd4ef77a54c8bc.jpg){:width 600}
	-

- ### Conclusion
	- In this hypothetical narrative, the creation and widespread adoption of cash-settled Bitcoin ETFs lead to a significant disjunction between the paper and physical Bitcoin markets, eventually resulting in a crisis of confidence and liquidity when the discrepancy becomes apparent. The repercussions would ripple through the cryptocurrency and traditional financial markets, leading to regulatory crackdowns, market crashes, and potentially a paradigm shift towards more decentralized and transparent financial systems. While this scenario is speculative, it underscores the importance of understanding and critically evaluating the implications of integrating cryptocurrency into traditional financial products and the systemic risks that could emerge from such integration.
- ![image.png](assets/image_1707899842471_0.png)
- <iframe width="100%" height="420" frameborder="0" src="https://www.theblock.co/data/crypto-markets/bitcoin-etf/spot-bitcoin-etf-assets/embed" title="Spot Bitcoin ETF AUM"></iframe>

- ### Regulatory Repercussions and Market Reaction
	- The revelation of a large-scale discrepancy between paper and physical Bitcoin would likely lead to intense regulatory scrutiny and possibly new regulations or bans on similar financial products. This regulatory response could stifle innovation and investment in the cryptocurrency space, at least temporarily, and lead to a loss of trust in financial institutions involved in the ETF market. The cryptocurrency community, known for its resilience and innovation, might respond by pushing further towards decentralized finance (DeFi) solutions and away from traditional financial systems. This shift could accelerate the adoption of technologies that provide more transparency and direct control over digital assets, such as improved self-custody solutions and transparent, decentralized exchanges that do not rely on traditional financial intermediaries.

- ### The Setup
	- Imagine a world where Bitcoin ETFs have gained significant traction, primarily among U.S. investors who prefer the regulated, traditional financial market entry points to the actual cryptocurrency. These ETFs are appealing because they offer exposure to Bitcoin's price movements without requiring investors to deal with the complexities and security concerns of holding the cryptocurrency. However, as these are cash-settled, the ETFs do not impact the actual supply and demand of Bitcoin directly but rather create a parallel market for Bitcoin exposure.
