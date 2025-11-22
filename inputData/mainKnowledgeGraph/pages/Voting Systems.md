- ### OntologyBlock
    - term-id:: BC-0495
    - preferred-term:: Voting Systems
    - ontology:: true
    - is-subclass-of:: [[DisruptiveTechnology]]

## Voting Systems

Voting Systems refers to blockchain-based electoral systems employing cryptographic verification, end-to-end verifiability, and distributed ledger technology to enable secure voting whilst facing critical security challenges identified by mit research showing vulnerabilities allowing vote alteration, academic consensus of "nearly universal" agreement that no technology can adequately secure online public elections, and real-world implementations spanning voatz (80,000+ votes across 50+ elections), estonia (50%+ votes cast online), and moscow (65,000 voters) demonstrating both potential cost reductions from $7-25 to under $0.50 per vote and significant security concerns.

- Industry adoption and implementations
  - Blockchain voting platforms such as Voatz and Scytl have been used in limited trials, primarily for absentee and overseas voters, but large-scale public elections remain rare.
  - Estonia continues to operate its Internet voting system (i-Voting), which incorporates blockchain for vote logging, though the system is not fully decentralised and relies on national identity infrastructure.
  - Moscow and other Russian cities have conducted small-scale blockchain voting pilots, but these have faced criticism over transparency and auditability.
- Notable organisations and platforms
  - Voatz: Deployed in various US states for military and overseas voters, with reported use in over 50 elections and more than 80,000 votes cast.
  - Scytl: Provides e-voting solutions in Europe, including blockchain-based components for vote logging and audit trails.
  - Estonia’s i-Voting: Over 50% of votes in recent national elections have been cast online, with blockchain used for vote storage and integrity cheques.
- UK and North England examples where relevant
  - The UK has not adopted blockchain voting for national or local elections, but several pilot projects and academic studies have explored its feasibility.
  - In North England, universities in Manchester, Leeds, Newcastle, and Sheffield have participated in research collaborations on secure e-voting, often in partnership with government agencies and tech firms.
  - Local councils in Greater Manchester and West Yorkshire have trialled digital voting for internal elections and community consultations, though these have not yet incorporated blockchain technology.
- Technical capabilities and limitations
  - Blockchain voting can provide strong audit trails, tamper-evident logs, and decentralised verification, but it does not solve all security problems.
  - Key limitations include the vulnerability of voter devices to malware, the risk of denial-of-service attacks, and the difficulty of ensuring voter privacy and coercion resistance.
  - Cost reductions have been observed in some pilot projects, with estimates suggesting per-vote costs can drop from $7–25 to under $0.50, but these savings are context-dependent and may not scale to large public elections.
- Standards and frameworks
  - International standards such as ISO/IEC 27001 and NIST guidelines provide general security frameworks for e-voting systems.
  - The European Union has developed specific guidelines for blockchain-based voting, emphasising transparency, auditability, and voter verifiability.

## Technical Details

- **Id**: bc-0495-voting-systems-ontology
- **Collapsed**: true
- **Source Domain**: blockchain
- **Status**: complete
- **Public Access**: true
- **Authority Score**: 0.82
- **Maturity**: draft
- **Owl:Class**: bc:VotingSystems
- **Owl:Physicality**: ConceptualEntity
- **Owl:Role**: Concept
- **Belongstodomain**: [[BlockchainDomain]]
- **Blockchainrelevance**: High
- **Lastvalidated**: 2025-11-14

## Research & Literature

- Key academic papers and sources
  - Park, S., Specter, M., Narula, N., Rivest, R. L. (2021). Going from bad to worse: from Internet voting to blockchain voting. Journal of Cybersecurity, 7(1), tyaa025. https://doi.org/10.1093/cybsec/tyaa025
  - Shaikh, A., Adhikari, N., Nazir, A., Shah, A. S., Baig, S., Al Shihi, H. (2025). Blockchain-enhanced electoral integrity: a robust model for secure voting. F1000Research, 14, 223. https://doi.org/10.12688/f1000research.160087.3
  - Jefferson, D. (2023). The Myth of “Secure” Blockchain Voting. U.S. Vote Foundation. https://www.usvotefoundation.org/blockchain-voting-is-not-a-security-strategy
  - CoinLaw. (2025). Blockchain in Voting Systems Statistics 2025. https://coinlaw.io/blockchain-in-voting-systems-statistics/
  - ACM Digital Library. (2025). A Comprehensive Analysis of Blockchain-Based Voting Systems. https://dl.acm.org/doi/10.1145/3723178.3723275
  - SSRN. (2025). Blockchain-Based E-Voting Systems: A Systematic Literature Review. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5370817
- Ongoing research directions
  - Hybrid models combining blockchain with traditional paper ballots
  - Zero-knowledge proofs for voter privacy
  - Layer 2 solutions for scalability and performance
  - Threat modelling and real-world stress-testing of blockchain voting systems

## UK Context

- British contributions and implementations
  - The UK has been active in research on secure e-voting, with contributions from universities, government agencies, and independent think tanks.
  - The National Cyber Security Centre (NCSC) has published guidance on the risks and benefits of blockchain voting, emphasising the need for rigorous security testing and public scrutiny.
- North England innovation hubs (if relevant)
  - Universities in Manchester, Leeds, Newcastle, and Sheffield have established research groups focused on digital democracy and secure voting.
  - Collaborative projects with local councils and tech firms have explored the use of blockchain for internal elections and community consultations.
- Regional case studies
  - Greater Manchester Council has trialled digital voting for internal elections, with plans to expand to community consultations.
  - West Yorkshire Combined Authority has explored the use of blockchain for secure data sharing and audit trails in public services.

## Future Directions

- Emerging trends and developments
  - Increased use of hybrid models combining blockchain with traditional paper ballots
  - Greater emphasis on voter privacy and coercion resistance
  - Development of international standards and best practices for blockchain voting
- Anticipated challenges
  - Ensuring the security of voter devices
  - Mitigating the risk of denial-of-service attacks
  - Building public trust in new voting technologies
- Research priorities
  - Formal verification of blockchain voting protocols
  - Real-world stress-testing and threat modelling
  - Development of user-friendly interfaces for secure voting

## References

1. Park, S., Specter, M., Narula, N., Rivest, R. L. (2021). Going from bad to worse: from Internet voting to blockchain voting. Journal of Cybersecurity, 7(1), tyaa025. https://doi.org/10.1093/cybsec/tyaa025
2. Shaikh, A., Adhikari, N., Nazir, A., Shah, A. S., Baig, S., Al Shihi, H. (2025). Blockchain-enhanced electoral integrity: a robust model for secure voting. F1000Research, 14, 223. https://doi.org/10.12688/f1000research.160087.3
3. Jefferson, D. (2023). The Myth of “Secure” Blockchain Voting. U.S. Vote Foundation. https://www.usvotefoundation.org/blockchain-voting-is-not-a-security-strategy
4. CoinLaw. (2025). Blockchain in Voting Systems Statistics 2025. https://coinlaw.io/blockchain-in-voting-systems-statistics/
5. ACM Digital Library. (2025). A Comprehensive Analysis of Blockchain-Based Voting Systems. https://dl.acm.org/doi/10.1145/3723178.3723275
6. SSRN. (2025). Blockchain-Based E-Voting Systems: A Systematic Literature Review. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5370817

## Metadata

- **Migration Status**: Ontology block enriched on 2025-11-12
- **Last Updated**: 2025-11-12
- **Review Status**: Comprehensive editorial review
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable
