# DAO Governance for Telecollaboration

- ### OntologyBlock
  id:: dao-governance-telecollaboration-ontology
  collapsed:: true
  - ontology:: true
    - is-subclass-of:: [[TelecollaborationTechnology]]
  - term-id:: TELE-252
  - preferred-term:: DAO Governance for Telecollaboration
  - alternate-terms::
  - Decentralised Autonomous Organisation Governance
  - Blockchain Governance for Remote Teams
  - Token-Based Collaboration Governance
  - source-domain:: tele
  - status:: active
  - public-access:: true
  - definition:: "The application of decentralised autonomous organisation (DAO) governance mechanisms—token-weighted voting, proposal systems, treasury management—to coordinate geographically distributed teams through on-chain decision-making, enabling democratic, transparent collaboration without centralised management or hierarchical authority structures."
  - maturity:: developing
  - authority-score:: 0.79
  - owl:class:: tele:DAOGovernanceTelecollaboration
  - owl:physicality:: ConceptualEntity
  - owl:role:: Process
  - belongsToDomain::
  - [[TELE-0000-telepresence-domain]]
  - [[BlockchainCollaboration]]
  - bridges-to::
  - [[BlockchainDomain]]


## Definition

**DAO Governance for Telecollaboration** enables distributed teams to make decisions democratically through blockchain-based voting. Token holders propose initiatives ("Fund project X with £50K"), vote proportionally to token ownership or reputation, and approved proposals execute automatically via smart contracts [[TELE-251-smart-contract-coordination]].

## Governance Models

- **Token-Weighted Voting**: 1 token = 1 vote (risk: plutocracy)
- **Quadratic Voting**: Cost of N votes = N² tokens (prevents whale dominance)
- **Reputation-Based**: Non-transferable reputation from contributions

## Examples

- **Uniswap DAO**: 400K token holders govern $5B decentralised exchange
- **Gitcoin**: Funds open-source developers via quadratic funding
- **MakerDAO**: Manages $5B DeFi protocol via MKR token voting

## Advantages

- **Democratic**: All token holders participate in decisions
- **Transparent**: Proposals, votes, transactions public on blockchain
- **Global**: Anyone with internet can join, regardless of geography/citizenship

## Challenges

- **Low Turnout**: Only 5-10% of token holders vote (governance fatigue)
- **Plutocracy**: Wealthy token holders dominate decisions
- **Slow**: Proposal-vote-execution cycle takes days-weeks

## Related Concepts

- [[TELE-002-telecollaboration]]
- [[TELE-250-blockchain-collaboration]]
- [[TELE-251-smart-contract-coordination]]
- [[DecentralisedAutonomousOrganisation]]

## Metadata

- **Term-ID**: TELE-252
- **Last Updated**: 2025-11-16
- **Maturity**: Developing
- **Authority Score**: 0.79
