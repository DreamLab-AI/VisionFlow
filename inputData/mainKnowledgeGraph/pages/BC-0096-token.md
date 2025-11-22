- ### OntologyBlock
  id:: token-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: BC-0096
    - preferred-term:: Token
    - source-domain:: blockchain
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-10-28

  - **Definition**
    - definition:: A token is a digital asset representation on a blockchain that confers specific rights, utility, or value to its holder, implemented as a cryptographically-secured unit that can be owned, transferred, and programmably controlled through smart contracts according to defined rules and protocols.
    - maturity:: mature
    - source:: [[ISO/IEC 23257:2021]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: bc:Token
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
    - owl:inferred-class:: bc:VirtualObject
    - belongsToDomain:: [[TokenEconomicsDomain]]
    - implementedInLayer:: [[EconomicLayer]]

  - #### Relationships
    id:: token-relationships
    - is-subclass-of:: [[Digital Asset]], [[Blockchain Entity]], [[Transferable Right]]

  - #### OWL Axioms
    id:: token-owl-axioms
    collapsed:: true
    - ```clojure
      Prefix(:=<http://metaverse-ontology.org/blockchain#>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)

Ontology(<http://metaverse-ontology.org/blockchain/BC-0096>

  ## Class Declaration
  Declaration(Class(:Token))

  ## Subclass Relationships
  SubClassOf(:Token :DigitalAsset)
  SubClassOf(:Token :BlockchainEntity)
  SubClassOf(:Token :TransferableRight)

  ## Essential Properties
  SubClassOf(:Token
    (ObjectExactCardinality 1 :deployedOn :Blockchain))

  SubClassOf(:Token
    (ObjectExactCardinality 1 :conformsTo :TokenStandard))

  SubClassOf(:Token
    (ObjectSomeValuesFrom :hasOwner :Address))

  SubClassOf(:Token
    (ObjectSomeValuesFrom :enablesOperation :TokenOperation))

  SubClassOf(:Token
    (DataHasValue :isTransferable "true"^^xsd:boolean))

  ## Token Properties
  SubClassOf(:Token
    (DataSomeValuesFrom :hasTotalSupply xsd:decimal))

  SubClassOf(:Token
    (DataSomeValuesFrom :hasDecimals xsd:nonNegativeInteger))

  SubClassOf(:Token
    (ObjectSomeValuesFrom :implementedBy :SmartContract))

  ## Data Properties
  DataPropertyAssertion(:tokenName :Token xsd:string)
  DataPropertyAssertion(:tokenSymbol :Token xsd:string)
  DataPropertyAssertion(:totalSupply :Token xsd:decimal)
  DataPropertyAssertion(:decimals :Token xsd:nonNegativeInteger)
  DataPropertyAssertion(:isMinDurable :Token xsd:boolean)
  DataPropertyAssertion(:isBurnable :Token xsd:boolean)

  ## Object Properties
  ObjectPropertyAssertion(:ownedBy :Token :Address)
  ObjectPropertyAssertion(:transferredTo :Token :Address)
  ObjectPropertyAssertion(:governedBy :Token :TokenGovernance)
  ObjectPropertyAssertion(:representsRight :Token :Right)

  ## Annotations
  AnnotationAssertion(rdfs:label :Token "Token"@en)
  AnnotationAssertion(rdfs:comment :Token
    "Digital asset representation on blockchain with transferable rights"@en)
  AnnotationAssertion(:termID :Token "BC-0096")

  ## Token Type Subclasses
  Declaration(Class(:FungibleToken))
  SubClassOf(:FungibleToken :Token)
  SubClassOf(:FungibleToken
    (DataHasValue :isFungible "true"^^xsd:boolean))

  Declaration(Class(:NonFungibleToken))
  SubClassOf(:NonFungibleToken :Token)
  SubClassOf(:NonFungibleToken
    (DataHasValue :isFungible "false"^^xsd:boolean))

  Declaration(Class(:SemiFungibleToken))
  SubClassOf(:SemiFungibleToken :Token)

  ## Disjoint Classes
  DisjointClasses(:FungibleToken :NonFungibleToken)
)
      ```

- ## Next Word/Token Prediction
	- **Method:** This technique unlocks the entire body of human text as a potential training dataset. Imagine having access to every book, article, blog post, and conversation ever written.
	- **Process:** The AI system learns by predicting the next word in a sequence. It takes into account the preceding words, trying to decipher the underlying grammar, semantics, and even the author's style.
	- **Process:** Instead of relying on labelled datasets, we intentionally degrade images by adding noise. This could be random pixels, blurring, or other forms of distortion. The AI's task is to learn to reverse this degradation, reconstructing the original, pristine image.
	- **Concept:** By learning to remove noise, the AI learns to identify the fundamental features and patterns within an image. It becomes better at distinguishing between real details and random noise. This process is similar to how our own brains filter out distractions to focus on relevant information.
	- **Significance:** This approach opens up a vast new world of possibilities for training AI systems, enabling them to learn from unlabeled image data.
		- **Tokens:** Words or image fragments are represented as numerical embeddings, called tokens. These tokens allow the AI to process language and images in a numerical way.
		- **Neurons:** Neurons are the individual nodes within the AI network. They are typically organised into layers, with each layer performing a specific task.
		- **Activations:** The values at specific neurons are called activations. These activations indicate how active a particular neuron is, providing insights into the AI's internal decision-making process.
	- **Significance:** This example demonstrates that AI systems can learn complex concepts like sentiment without explicit instructions. They learn to represent these concepts internally, using them to perform their primary task more effectively.
	- **Concept:** This idea, of semantics (meaning) emerging from a syntactic (structure) process, is crucial to understanding the power of AI.
- #### Ether Ultra Hard Money Narrative
	- Part of the challenge Ethereum faces is wrapped up with its complex token emission schedule. This is the rate at which tokens are generated and ‘burnt’ or destroyed in the network. The total supply of tokens is uncertain, and both emission and burn schedules are regularly tinkered with by the project. The changes to the rate at which ETH are generated.
	  ![Image](./assets/3fe8a20a55cfa025f4f59f7b04483196d7f28708.png)
	- The rate of token generation has changed unpredictably over time. Rights requested.
	- In addition, a recent upgrade (EIP-1559) results in tokens now being burnt at a higher rate than they are produced, deliberately leading to a diminishing supply. In theory, this increases the value of each ETH on the network at around 1% per year. It’s very complex, with impacts on transaction fees, waiting time, and consensus security, as examined by Liu et al.[[liu2022empirical]]. Additionally, there is now talk (by [Buterin](https://time.com/6158182/vitalik-buterin-ethereum-profile/), the creator of Ethereum) of extending this burn mechanism [further into the network](https://ethresear.ch/t/multidimensional-eip-1559/11651).
- Ethereum was designed from the beginning to move to a ‘proof of stake’ model where token holders underpin network consensus through complex automated voting systems based upon their token holding. This is now called [Ethereum Consensus Layer](https://blog.ethereum.org/2022/01/24/the-great-eth2-renaming/). This recent ‘Merge’ upgrade has reduced the carbon footprint of the network, a laudable thing, though it seems the GPUs and data centres have just gone on to be elsewhere. It has not lowered the cost to users nor improved performance. As part of the switching roadmap, users were asked to lock up 32 ETH tokens each (a substantial allocation of capital). In total, there are around 14 million of these tokens, and it is those users who now control the network. This money is likely stuck on the network until at least 2024, a significant delay when compared to the original promises.
- This means that proof of stake has problems in that the majority owners ‘decide’ the truth of the chain to a degree and must by design have the ability to override prior consensus choices. Remember that these users are now trapped in their positions. Four major entities now control the rules of the chain and have already agreed to censor certain banned addresses. Proof of stake is probably inherently broken[[poelstra2015stake]]. This has [f](https://notes.ethereum.org/@djrtwo/risks-of-lsd)or malicious actors who have sufficient control of the existing history of the chain, thought to be [in the region of $50M](https://twitter.com/MTorgin/status/1521433474820890624)[[mackinga2022twap]]. Like much of the rest of ‘crypto’, the proposed changes will concentrate decisions and economic rewards in the hands of major players, early investors, and incumbents.
- ![image.png](assets/image_1742487476628_0.png)
- This is a far cry from the stated aims of the technology. The move to proof of stake has recently earned it the [MIT breakthrough technology award](https://www.technologyreview.com/2022/02/23/1044960/proof-of-stake-cryptocurrency/), despite not being complete (validators cannot yet sell their voting stakes). It’s clearly a technology that is designed to innovate at the expense of predictability. This might work out very well for the platform, but right now the barrier to participation (in gas fees) is so high that we do not intend for Ethereum to be in scope as a method for value transfer within metaverses.
- #### Web5, Bluesky, & Microsoft ION
- Promisingly Jack Dorsey’s company TBD is working on a project [called“Web5”](https://developer.tbd.website/projects/web5/). Details are scantbut the promise is decentralised and/or self hosted data and identityrunning on Bitcoin, without recourse to a new token. it“Componentsinclude decentralized identifiers (DIDs), decentralized web node (DWNs),self-sovereign identity service (SSIS) and a self-sovereign identitysoftware development kit (ssi-sdk)”.
- Web5 leverages the ION identity stack. All this looks to be exactly whatour metaverse system requires, but the complexity is likely to be quitehigh as it is to be built on existing DID/SSI research which is prettycomplex and perhaps has problems.
- They readily admit they [do not have a workingsolution](https://atproto.com/guides/identity) at this time: it“Atpresent, none of the DID methods meet our standards fully. Many existingDID networks are permissionless blockchains which achieve the abovegoals but with relatively poor latency (ION takes roughly 20 minutes forcommitment finality). Therefore we have chosen to support did-web and atemporary method we’ve created called did-placeholder. We expect thissituation to evolve as new solutions emerge.”
- ## High level analysis
	- It seems possible that eight value propositions are therefore emerging:
	  id:: 661d5f6a-ce5e-479e-8722-2128890607bd
		- Bitcoin the speculative asset (or greater fool bubble [66]). Nations such as the USA, who own 30% of the asset have bid up the price of the tokens during a period of very cheap money, and this has led to a high valuation for the tokens, with a commensurately high network security through the hash rate (mining). This could be a speculative bubble, with the asset shifting to one of the other valuations below. There is more on this subject in the money section later.
		- Gambling, the "Financial nihilism use case", is well explained by Travis Kling in a twitter thread. "Number go up" is clearly the predominant use case at this time for both Bitcoin and crypto. Kling's analysis paints a vivid picture of the current socio-economic climate, where financial nihilism—stemming from stifling cost of living, dwindling upward mobility, and an untenable ratio of median home prices to median income—fuels speculative gambling within the crypto space. This atmosphere encourages individuals to invest in highly speculative assets with the slim hope of substantial returns, akin to purchasing lottery tickets. In this context, Bitcoin and other cryptocurrencies become vehicles for extreme risk-taking, driven not by a belief in their fundamental value but by the desperation and desire for a quick financial win in a system perceived as increasingly rigged against the average person. Kling's observations suggest that for many, the gamble on cryptocurrencies is less about informed investment and more about the desperate swing for the fences, embodying a form of financial nihilism that sees traditional avenues of wealth accumulation as blocked or insufficient. This speculative gamble is further fuelled by the allure of significant gains, regardless of the inherent risks or the long-term sustainability of such investments.
		- [twitter link to the render loading below](https://twitter.com/Travis_Kling/status/1753455596462878815)
		  {{twitter https://twitter.com/Travis_Kling/status/1753455596462878815}}
		- Bitcoin the (human) monetary network, and ‘emerging market’ value transfer mechanism. This will be most useful for Africa (especially Nigeria), India, and South America. There is no sense of the “value” of this network at this time, but it’s the aspect we need for our collaborative mixed reality application. For this use the price must simply be high enough to ensure that mining viably secures the network. This security floor is unfortunately a ‘known unknown’. If a global Bitcoin monetary axis evolves (as in the Money chapter later) the network would certainly require a higher rate than currently, suggestive of a higher price of token to ensure mining [67].
		- Bitcoin as an autonomous AI monetary network. In an era where AI actors perform tasks on behalf of humans in digital realms such as cyberspace, these AI actors will require a reliable and efficient means of transaction. AI agents can perform, transact and negotiate, and execute work contracts in near real-time. For this use, the primary requirement is not a high token price, but rather a high level of network security and scalability that can support an enormous volume of transactions.
		- The Lightning Network of Bitcoin might be a starting point but the robustness of the system, against potential AI exploits, is yet to be confirmed. As AI systems become more complex and autonomous, there is an increasing need for decentralized AI governance mechanisms that can prevent the concentration of power and ensure ethical AI development and deployment. Bitcoin can serve as a basis for this, providing a decentralized, transparent, and immutable record of AI decisions and actions. Furthermore, Bitcoin’s proof-of-work consensus mechanism could potentially be adapted to enforce AI adherence to agreed-upon rules or norms. In this context, Bitcoin’s value extends beyond its token price and into its potential contributions to AI governance and ethics.
		- This is Bitcoin as an AI economy. It's notable that scaling solutions like [[Cashu]] and or [[RGB and Client Side Validation]] are likely required in addition to more established technologies like [[Lightning and Similar L2]]; this technical landscape isn't quite ready. -
		- Bitcoin as a hedge against future quantum computation. It has been argued that the advent of quantum computers could threaten the security of many existing cryptographic systems. Bitcoin’s open-source nature allows for the integration of post-quantum cryptographic algorithms, safeguarding it against quantum threats. In this sense, investment in Bitcoin might also be seen as an investment in a future-proof monetary network. This assertion depends on the assumption that Bitcoin’s protocol will adapt in time to incorporate such cryptographic advances before quantum computing becomes a real threat to its integrity. The practical implementation of these technologies might see a shift in the network’s dynamics, the hash rate, mining cost, and token value.
		- Bitcoin’s value in terms of ‘sunk opportunity cost’. This refers to the value that could have been generated if the resources invested in a particular activity had been utilised elsewhere. In the context of Bitcoin, this includes the investments made in mining equipment, power, facilities, and the hiring of skilled personnel to maintain the operations. The sunk opportunity cost of Bitcoin can be substantial. It can be argued that the value of Bitcoin must take this cost into consideration, as the resources could have been allocated to other productive sectors or investments [68]. Of course, there remains the infamous sunk cost fallacy, which refers to the tendency of individuals or organizations to continue investing in a project or decision based on the amount of resources already spent, rather than evaluating the current and future value of the investment. This indeed tends to lead to a cyclical boom and bust dynamic in the industrial mining communities. The ultimate fallacy would occur if miners or investors continued to invest in mining equipment and operations solely because of the resources that have already been spent on them, and the asset simply crashes to nothing from here. It’s a shaky justification because it assumes the future is the same as the past. -
		- Bitcoin as a flexible load in power distribution systems, and methane mitigation ‘asset’, and ‘subsidised heater’ for varied applications such as growing and drying. Again there is no price against this, but we can perhaps grossly estimate it at around half the current hash rate if 50% of the network is currently green energy. This would imply a price for the asset roughly where it is now (ie, not orders of magnitude higher or lower). -
		- The 2023 global bank runs have awoken some companies to the risks of access to cash flows in a potential crisis [69]. Access to a small cache (in corporate treasury terms) of a highly liquid & tradable asset could allow continuity of payroll in a ‘24/7’ global context. This could avoid or at least mitigate the panic which ensues in companies when banks are forces to suddenly wind up their operations.
		- Amusingly Ben Hunt suggests in an online article that the true value of Bitcoin can be couched in terms of it’s value simply as ‘art’. He posits that at this time the narrative is simply so seductive and powerful that people (being people) are choosing to value their involvement in the economics of the space as they might a work of art. It’s a fascinating idea, and intuitively, probably it’s right.
- #### Ether Ultra Hard Money Narrative
	- Part of the challenge Ethereum faces is wrapped up with its complex token emission schedule. This is the rate at which tokens are generated and ‘burnt’ or destroyed in the network. The total supply of tokens is uncertain, and both emission and burn schedules are regularly tinkered with by the project. The changes to the rate at which ETH are generated.
	  ![Image](./assets/3fe8a20a55cfa025f4f59f7b04483196d7f28708.png)
	- The rate of token generation has changed unpredictably over time. Rights requested.
	- In addition, a recent upgrade (EIP-1559) results in tokens now being burnt at a higher rate than they are produced, deliberately leading to a diminishing supply. In theory, this increases the value of each ETH on the network at around 1% per year. It’s very complex, with impacts on transaction fees, waiting time, and consensus security, as examined by Liu et al.[[liu2022empirical]]. Additionally, there is now talk (by [Buterin](https://time.com/6158182/vitalik-buterin-ethereum-profile/), the creator of Ethereum) of extending this burn mechanism [further into the network](https://ethresear.ch/t/multidimensional-eip-1559/11651).
- Ethereum was designed from the beginning to move to a ‘proof of stake’ model where token holders underpin network consensus through complex automated voting systems based upon their token holding. This is now called [Ethereum Consensus Layer](https://blog.ethereum.org/2022/01/24/the-great-eth2-renaming/). This recent ‘Merge’ upgrade has reduced the carbon footprint of the network, a laudable thing, though it seems the GPUs and data centres have just gone on to be elsewhere. It has not lowered the cost to users nor improved performance. As part of the switching roadmap, users were asked to lock up 32 ETH tokens each (a substantial allocation of capital). In total, there are around 14 million of these tokens, and it is those users who now control the network. This money is likely stuck on the network until at least 2024, a significant delay when compared to the original promises.
- This means that proof of stake has problems in that the majority owners ‘decide’ the truth of the chain to a degree and must by design have the ability to override prior consensus choices. Remember that these users are now trapped in their positions. Four major entities now control the rules of the chain and have already agreed to censor certain banned addresses. Proof of stake is probably inherently broken[[poelstra2015stake]]. This has [f](https://notes.ethereum.org/@djrtwo/risks-of-lsd)or malicious actors who have sufficient control of the existing history of the chain, thought to be [in the region of $50M](https://twitter.com/MTorgin/status/1521433474820890624)[[mackinga2022twap]]. Like much of the rest of ‘crypto’, the proposed changes will concentrate decisions and economic rewards in the hands of major players, early investors, and incumbents.
- ![image.png](assets/image_1742487476628_0.png)
- This is a far cry from the stated aims of the technology. The move to proof of stake has recently earned it the [MIT breakthrough technology award](https://www.technologyreview.com/2022/02/23/1044960/proof-of-stake-cryptocurrency/), despite not being complete (validators cannot yet sell their voting stakes). It’s clearly a technology that is designed to innovate at the expense of predictability. This might work out very well for the platform, but right now the barrier to participation (in gas fees) is so high that we do not intend for Ethereum to be in scope as a method for value transfer within metaverses.
- #### Web5, Bluesky, & Microsoft ION
- Promisingly Jack Dorsey’s company TBD is working on a project [called“Web5”](https://developer.tbd.website/projects/web5/). Details are scantbut the promise is decentralised and/or self hosted data and identityrunning on Bitcoin, without recourse to a new token. it“Componentsinclude decentralized identifiers (DIDs), decentralized web node (DWNs),self-sovereign identity service (SSIS) and a self-sovereign identitysoftware development kit (ssi-sdk)”.
- Web5 leverages the ION identity stack. All this looks to be exactly whatour metaverse system requires, but the complexity is likely to be quitehigh as it is to be built on existing DID/SSI research which is prettycomplex and perhaps has problems.
- They readily admit they [do not have a workingsolution](https://atproto.com/guides/identity) at this time: it“Atpresent, none of the DID methods meet our standards fully. Many existingDID networks are permissionless blockchains which achieve the abovegoals but with relatively poor latency (ION takes roughly 20 minutes forcommitment finality). Therefore we have chosen to support did-web and atemporary method we’ve created called did-placeholder. We expect thissituation to evolve as new solutions emerge.”
- ## High level analysis
	- It seems possible that eight value propositions are therefore emerging:
	  id:: 661d5f6a-ce5e-479e-8722-2128890607bd
		- Bitcoin the speculative asset (or greater fool bubble [66]). Nations such as the USA, who own 30% of the asset have bid up the price of the tokens during a period of very cheap money, and this has led to a high valuation for the tokens, with a commensurately high network security through the hash rate (mining). This could be a speculative bubble, with the asset shifting to one of the other valuations below. There is more on this subject in the money section later.
		- Gambling, the "Financial nihilism use case", is well explained by Travis Kling in a twitter thread. "Number go up" is clearly the predominant use case at this time for both Bitcoin and crypto. Kling's analysis paints a vivid picture of the current socio-economic climate, where financial nihilism—stemming from stifling cost of living, dwindling upward mobility, and an untenable ratio of median home prices to median income—fuels speculative gambling within the crypto space. This atmosphere encourages individuals to invest in highly speculative assets with the slim hope of substantial returns, akin to purchasing lottery tickets. In this context, Bitcoin and other cryptocurrencies become vehicles for extreme risk-taking, driven not by a belief in their fundamental value but by the desperation and desire for a quick financial win in a system perceived as increasingly rigged against the average person. Kling's observations suggest that for many, the gamble on cryptocurrencies is less about informed investment and more about the desperate swing for the fences, embodying a form of financial nihilism that sees traditional avenues of wealth accumulation as blocked or insufficient. This speculative gamble is further fuelled by the allure of significant gains, regardless of the inherent risks or the long-term sustainability of such investments.
		- [twitter link to the render loading below](https://twitter.com/Travis_Kling/status/1753455596462878815)
		  {{twitter https://twitter.com/Travis_Kling/status/1753455596462878815}}
		- Bitcoin the (human) monetary network, and ‘emerging market’ value transfer mechanism. This will be most useful for Africa (especially Nigeria), India, and South America. There is no sense of the “value” of this network at this time, but it’s the aspect we need for our collaborative mixed reality application. For this use the price must simply be high enough to ensure that mining viably secures the network. This security floor is unfortunately a ‘known unknown’. If a global Bitcoin monetary axis evolves (as in the Money chapter later) the network would certainly require a higher rate than currently, suggestive of a higher price of token to ensure mining [67].
		- Bitcoin as an autonomous AI monetary network. In an era where AI actors perform tasks on behalf of humans in digital realms such as cyberspace, these AI actors will require a reliable and efficient means of transaction. AI agents can perform, transact and negotiate, and execute work contracts in near real-time. For this use, the primary requirement is not a high token price, but rather a high level of network security and scalability that can support an enormous volume of transactions.
		- The Lightning Network of Bitcoin might be a starting point but the robustness of the system, against potential AI exploits, is yet to be confirmed. As AI systems become more complex and autonomous, there is an increasing need for decentralized AI governance mechanisms that can prevent the concentration of power and ensure ethical AI development and deployment. Bitcoin can serve as a basis for this, providing a decentralized, transparent, and immutable record of AI decisions and actions. Furthermore, Bitcoin’s proof-of-work consensus mechanism could potentially be adapted to enforce AI adherence to agreed-upon rules or norms. In this context, Bitcoin’s value extends beyond its token price and into its potential contributions to AI governance and ethics.
		- This is Bitcoin as an AI economy. It's notable that scaling solutions like [[Cashu]] and or [[RGB and Client Side Validation]] are likely required in addition to more established technologies like [[Lightning and Similar L2]]; this technical landscape isn't quite ready. -
		- Bitcoin as a hedge against future quantum computation. It has been argued that the advent of quantum computers could threaten the security of many existing cryptographic systems. Bitcoin’s open-source nature allows for the integration of post-quantum cryptographic algorithms, safeguarding it against quantum threats. In this sense, investment in Bitcoin might also be seen as an investment in a future-proof monetary network. This assertion depends on the assumption that Bitcoin’s protocol will adapt in time to incorporate such cryptographic advances before quantum computing becomes a real threat to its integrity. The practical implementation of these technologies might see a shift in the network’s dynamics, the hash rate, mining cost, and token value.
		- Bitcoin’s value in terms of ‘sunk opportunity cost’. This refers to the value that could have been generated if the resources invested in a particular activity had been utilised elsewhere. In the context of Bitcoin, this includes the investments made in mining equipment, power, facilities, and the hiring of skilled personnel to maintain the operations. The sunk opportunity cost of Bitcoin can be substantial. It can be argued that the value of Bitcoin must take this cost into consideration, as the resources could have been allocated to other productive sectors or investments [68]. Of course, there remains the infamous sunk cost fallacy, which refers to the tendency of individuals or organizations to continue investing in a project or decision based on the amount of resources already spent, rather than evaluating the current and future value of the investment. This indeed tends to lead to a cyclical boom and bust dynamic in the industrial mining communities. The ultimate fallacy would occur if miners or investors continued to invest in mining equipment and operations solely because of the resources that have already been spent on them, and the asset simply crashes to nothing from here. It’s a shaky justification because it assumes the future is the same as the past. -
		- Bitcoin as a flexible load in power distribution systems, and methane mitigation ‘asset’, and ‘subsidised heater’ for varied applications such as growing and drying. Again there is no price against this, but we can perhaps grossly estimate it at around half the current hash rate if 50% of the network is currently green energy. This would imply a price for the asset roughly where it is now (ie, not orders of magnitude higher or lower). -
		- The 2023 global bank runs have awoken some companies to the risks of access to cash flows in a potential crisis [69]. Access to a small cache (in corporate treasury terms) of a highly liquid & tradable asset could allow continuity of payroll in a ‘24/7’ global context. This could avoid or at least mitigate the panic which ensues in companies when banks are forces to suddenly wind up their operations.
		- Amusingly Ben Hunt suggests in an online article that the true value of Bitcoin can be couched in terms of it’s value simply as ‘art’. He posits that at this time the narrative is simply so seductive and powerful that people (being people) are choosing to value their involvement in the economics of the space as they might a work of art. It’s a fascinating idea, and intuitively, probably it’s right.
