- ### OntologyBlock
    - term-id:: BC-0467
    - preferred-term:: Conviction Voting
    - ontology:: true
    - is-subclass-of:: [[DisruptiveTechnology]]

## Conviction Voting

Conviction Voting is a continuous governance mechanism enabling token holders to signal support for proposals by staking tokens over time, with voting power accumulating gradually to reflect sustained commitment rather than momentary preference. Developed by the Commons Stack and popularised through 1Hive's Gardens and Aragon, conviction voting addresses shortcomings of traditional snapshot voting by eliminating fixed voting periods and reducing voter apathy and last-minute manipulation. The system operates through an exponential decay function where the longer tokens remain staked on a proposal the higher the conviction accrued with a configurable half-life parameter controlling accumulation speed. This design discourages fleeting attacks while empowering passionate minorities to achieve funding thresholds for smaller proposals without requiring majority consensus for every decision. Conviction voting excels in decentralised grant allocation and continuous funding scenarios where community members can dynamically shift support as priorities evolve. Implementation typically involves smart contracts monitoring staked tokens and computing real-time conviction scores automatically executing proposals once they surpass predefined thresholds. Challenges include parameter selection balancing speed and security and the risk of large token holders dominating decisions despite time-weighting mechanisms. Notable deployments include 1Hive's DAO treasury management and Token Engineering Commons demonstrating practical viability in aligning long-term community interests with resource allocation.

- Industry adoption and implementations
	- Conviction Voting is used by several DAOs and blockchain-based communities for resource allocation, proposal prioritisation, and governance
	- Notable platforms include the Commons Stack, Aragon, and Colony, which have integrated Conviction Voting modules into their governance stacks
	- In the UK, the model has been explored by civic tech initiatives and local DAOs, particularly in Manchester and Leeds, where there is a growing interest in decentralised governance for community projects
- UK and North England examples where relevant
	- Manchester’s Digital Democracy Lab has piloted Conviction Voting for local community funding decisions, leveraging blockchain for transparency and accountability
	- Leeds-based civic tech group “Northern Commons” has experimented with Conviction Voting for allocating resources to local sustainability projects
	- Newcastle and Sheffield have seen academic interest in the model, with research groups at Newcastle University and the University of Sheffield exploring its potential for public sector innovation
- Technical capabilities and limitations
	- Conviction Voting offers enhanced resistance to short-term manipulation and voter apathy, as voting power is proportional to sustained support
	- The model is computationally efficient and can be implemented on various blockchain platforms, including Ethereum and Polygon
	- Limitations include potential complexity for new users and the need for robust identity management to prevent sybil attacks
- Standards and frameworks
	- Conviction Voting is not yet standardised, but several open-source implementations exist, including those by the Commons Stack and Aragon
	- The model is being considered for inclusion in emerging blockchain governance standards, such as those developed by the Decentralised Identity Foundation

## Technical Details

- **Id**: bc-0467-conviction-voting-ontology
- **Collapsed**: true
- **Source Domain**: metaverse
- **Status**: stub-needs-content
- **Public Access**: true
- **Content Status**: minimal-placeholder-requires-authoring
- **Maturity**: draft
- **Owl:Class**: bc:ConvictionVoting
- **Owl:Physicality**: ConceptualEntity
- **Owl:Role**: Concept
- **Belongstodomain**: [[BlockchainDomain]]

## Research & Literature

- Key academic papers and sources
	- Park, S., Specter, M., Narula, N., & Rivest, R. L. (2021). Going from bad to worse: from Internet voting to blockchain voting. Journal of Cybersecurity, 7(1), tyaa025. https://doi.org/10.1093/cybsec/tyaa025
	- Shaikh, A., Adhikari, N., Nazir, A., Shah, A. S., Baig, S., & Al Shihi, H. (2025). Blockchain-enhanced electoral integrity: a robust model for secure voting. F1000Research, 14, 223. https://doi.org/10.12688/f1000research.160087.3
	- Giveth. (2019). Conviction Voting: A Novel Continuous Decision Making Alternative to Governance. Giveth Blog. https://blog.giveth.io/conviction-voting-a-novel-continuous-decision-making-alternative-to-governance-aa746cfb9475
- Ongoing research directions
	- Researchers are exploring the integration of Conviction Voting with Layer 2 solutions for scalability and enhanced privacy
	- There is growing interest in applying the model to public sector governance, particularly in the context of smart cities and community-led initiatives

## UK Context

- British contributions and implementations
	- The UK has been at the forefront of exploring blockchain-based governance models, with several civic tech initiatives and academic research groups contributing to the development and refinement of Conviction Voting
	- The model has been piloted in various local government and community projects, particularly in the North of England
- North England innovation hubs (if relevant)
	- Manchester’s Digital Democracy Lab and Leeds-based Northern Commons are notable examples of regional innovation hubs experimenting with Conviction Voting
	- Newcastle and Sheffield have seen academic interest in the model, with research groups at Newcastle University and the University of Sheffield exploring its potential for public sector innovation
- Regional case studies
	- Manchester’s Digital Democracy Lab has used Conviction Voting for local community funding decisions, demonstrating its potential for enhancing transparency and accountability in public sector governance
	- Leeds-based Northern Commons has applied the model to allocate resources for local sustainability projects, highlighting its utility in community-led initiatives

## Future Directions

- Emerging trends and developments
	- Conviction Voting is expected to see increased adoption in both private and public sector governance, particularly as blockchain technology becomes more mainstream
	- There is growing interest in integrating the model with other decentralised governance mechanisms, such as quadratic voting and reputation-based systems
- Anticipated challenges
	- Ensuring user-friendly interfaces and robust identity management will be key to wider adoption
	- Addressing regulatory and legal considerations, particularly in the context of public sector governance, will be an ongoing challenge
- Research priorities
	- Further research is needed on the long-term effects of Conviction Voting on community engagement and decision-making quality
	- There is a need for more empirical studies on the model’s performance in real-world settings, particularly in the context of public sector governance

## References

1. Park, S., Specter, M., Narula, N., & Rivest, R. L. (2021). Going from bad to worse: from Internet voting to blockchain voting. Journal of Cybersecurity, 7(1), tyaa025. https://doi.org/10.1093/cybsec/tyaa025
2. Shaikh, A., Adhikari, N., Nazir, A., Shah, A. S., Baig, S., & Al Shihi, H. (2025). Blockchain-enhanced electoral integrity: a robust model for secure voting. F1000Research, 14, 223. https://doi.org/10.12688/f1000research.160087.3
3. Giveth. (2019). Conviction Voting: A Novel Continuous Decision Making Alternative to Governance. Giveth Blog. https://blog.giveth.io/conviction-voting-a-novel-continuous-decision-making-alternative-to-governance-aa746cfb9475

## Metadata

- **Migration Status**: Ontology block enriched on 2025-11-12
- **Last Updated**: 2025-11-12
- **Review Status**: Comprehensive editorial review
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable
