- ### OntologyBlock
    - term-id:: AI-0382
    - preferred-term:: Fairness Constraints
    - ontology:: true
    - version:: 1.0


### Relationships
- is-subclass-of:: [[AlgorithmicBias]]

## Fairness Constraints

Fairness Constraints refers to fairness constraints are mathematical formalizations of equitable treatment in ai systems, expressed as conditions that predictions must satisfy relative to protected attributes. these constraints are categorised into three fundamental types based on independence criteria: independence (demographic parity) requires predictions to be independent of protected attributes (ŷ ⊥ a), meaning p(ŷ|a=0) = p(ŷ|a=1); separation (equalized odds) requires predictions to be independent of protected attributes conditional on true labels (ŷ ⊥ a | y), ensuring equal true positive and false positive rates across groups; and sufficiency (predictive parity) requires true labels to be independent of protected attributes conditional on predictions (y ⊥ a | ŷ), ensuring equal precision and calibration across groups. these constraints formalize fairness concepts like demographic parity, equalized odds, equal opportunity (separation for positive class only), and calibration into optimization problems during model training. however, impossibility theorems (chouldechova 2017, kleinberg et al. 2017) prove that when base rates differ between groups, certain combinations of fairness constraints cannot be simultaneously satisfied, necessitating context-dependent tradeoffs. implementation typically involves constrained optimization with lagrange multipliers, where accuracy loss is balanced against fairness violations through tunable regularization parameters, as formalized in foundational research by hardt et al. (2016) and barocas et al. (2019).

- Industry adoption and implementations
	- Fairness constraints are now routinely incorporated into AI development pipelines across sectors including finance, healthcare, criminal justice, and employment
	- Major platforms such as Fairlearn, IBM AI Fairness 360, and Google’s What-If Tool provide accessible frameworks for implementing and auditing fairness constraints
	- In the UK, organisations like the Alan Turing Institute and the Centre for Data Ethics and Innovation have published guidance and tools for fairness-aware AI development
- Notable organisations and platforms
	- Fairlearn (Microsoft Research)
	- AI Fairness 360 (IBM)
	- What-If Tool (Google)
	- LangChain, AutoGen, and CrewAI for workflow integration
- UK and North England examples where relevant
	- The Greater Manchester Combined Authority has piloted fairness-aware algorithms in public service delivery, focusing on equitable access to social care and housing
	- Leeds City Council has collaborated with local universities to audit AI-driven recruitment tools for bias, ensuring compliance with the Equality Act 2010
	- Newcastle University’s Centre for Social Justice and Community Action has developed fairness constraints for predictive policing models, aiming to reduce disproportionate targeting of minority communities
	- Sheffield Hallam University’s Advanced Manufacturing Research Centre has explored fairness in AI-driven workforce planning, with a focus on inclusive hiring practices
- Technical capabilities and limitations
	- Modern fairness constraints can enforce various notions of fairness, such as demographic parity, equalised odds, and counterfactual fairness
	- However, there are inherent limitations, including trade-offs between fairness and accuracy, the need for sensitive data, and the difficulty of hard-coding nuanced ethical principles
	- Some critics argue that technical solutions alone cannot fully address systemic discrimination, and that fairness constraints should be complemented by broader organisational and societal reforms
- Standards and frameworks
	- The UK’s Information Commissioner’s Office (ICO) has issued guidance on AI and data protection, emphasising the importance of fairness in automated decision-making
	- The European Union’s AI Act (2024) includes provisions for fairness and non-discrimination, influencing UK regulatory approaches post-Brexit
	- Industry standards such as ISO/IEC 23894 (2023) on AI risk management provide a framework for integrating fairness constraints into AI governance

## Technical Details

- **Id**: 0382-fairness-constraints-about
- **Collapsed**: true
- **Domain Prefix**: AI
- **Sequence Number**: 0382
- **Filename History**: ["AI-0382-fairness-constraints.md"]
- **Public Access**: true
- **Source Domain**: ai
- **Status**: in-progress
- **Last Updated**: 2025-10-29
- **Maturity**: mature
- **Source**: [[Hardt et al. (2016)]], [[Barocas et al. (2019)]], [[Chouldechova (2017)]]
- **Authority Score**: 0.95
- **Owl:Class**: aigo:FairnessConstraints
- **Owl:Physicality**: VirtualEntity
- **Owl:Role**: Process
- **Owl:Inferred Class**: aigo:VirtualProcess
- **Belongstodomain**: [[AIEthicsDomain]]
- **Implementedinlayer**: [[ConceptualLayer]]

## Research & Literature

- Key academic papers and sources
	- Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference (pp. 214–226). ACM. https://doi.org/10.1145/2090236.2090255
	- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In Advances in Neural Information Processing Systems (pp. 3315–3323). https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html
	- Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys, 54(6), 1–35. https://doi.org/10.1145/3457607
	- Selbst, A. D., Boyd, D., Friedler, S. A., Venkatasubramanian, S., & Vertesi, J. (2019). Fairness and abstraction in sociotechnical systems. In Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 59–68). ACM. https://doi.org/10.1145/3287560.3287598
	- Cowls, J., Tsamados, A., Taddeo, M., & Floridi, L. (2021). The AI fairness paradox: Why fairness can’t be automated. Philosophy & Technology, 34(4), 1087–1107. https://doi.org/10.1007/s13347-021-00458-4
- Ongoing research directions
	- Developing fairness constraints that are robust to distributional shifts and adversarial attacks
	- Exploring the intersection of fairness with privacy, transparency, and accountability
	- Investigating the role of fairness constraints in generative AI and large language models
	- Addressing the challenge of operationalising fairness in real-world, high-stakes decision-making contexts

## UK Context

- British contributions and implementations
	- The UK has been at the forefront of AI ethics and fairness research, with institutions such as the Alan Turing Institute, the Ada Lovelace Institute, and the Centre for Data Ethics and Innovation leading the way
	- The government’s AI Council has published a national strategy for AI, which includes a strong emphasis on fairness and inclusivity
	- The ICO’s guidance on AI and data protection has helped shape best practices for fairness-aware AI development in the UK
- North England innovation hubs (if relevant)
	- Manchester’s Digital Innovation Factory has hosted workshops and hackathons focused on fairness in AI, bringing together academics, industry experts, and community stakeholders
	- Leeds has established a Centre for Responsible AI, which collaborates with local businesses and public sector organisations to develop and deploy fairness-aware AI solutions
	- Newcastle’s Centre for Social Justice and Community Action has worked with local authorities to audit and improve the fairness of AI-driven public services
	- Sheffield’s Advanced Manufacturing Research Centre has explored the use of fairness constraints in workforce planning and recruitment, with a focus on inclusive hiring practices
- Regional case studies
	- The Greater Manchester Combined Authority’s use of fairness-aware algorithms in social care allocation has been cited as a model for other UK cities
	- Leeds City Council’s collaboration with the University of Leeds on AI-driven recruitment tools has led to improved diversity and inclusion in local government hiring
	- Newcastle University’s work on predictive policing models has informed national debates about the role of fairness constraints in law enforcement

## Future Directions

- Emerging trends and developments
	- Increasing use of fairness constraints in generative AI and large language models
	- Growing interest in explainable and interpretable fairness constraints
	- Development of fairness-aware AI governance frameworks and regulatory standards
- Anticipated challenges
	- Balancing fairness with other desirable properties such as accuracy, efficiency, and privacy
	- Addressing the limitations of technical solutions in the face of systemic discrimination
	- Ensuring that fairness constraints are accessible and usable by non-expert practitioners
- Research priorities
	- Developing fairness constraints that are robust to distributional shifts and adversarial attacks
	- Exploring the intersection of fairness with privacy, transparency, and accountability
	- Investigating the role of fairness constraints in real-world, high-stakes decision-making contexts

## References

1. Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference (pp. 214–226). ACM. https://doi.org/10.1145/2090236.2090255
2. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. In Advances in Neural Information Processing Systems (pp. 3315–3323). https://proceedings.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html
3. Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. ACM Computing Surveys, 54(6), 1–35. https://doi.org/10.1145/3457607
4. Selbst, A. D., Boyd, D., Friedler, S. A., Venkatasubramanian, S., & Vertesi, J. (2019). Fairness and abstraction in sociotechnical systems. In Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 59–68). ACM. https://doi.org/10.1145/3287560.3287598
5. Cowls, J., Tsamados, A., Taddeo, M., & Floridi, L. (2021). The AI fairness paradox: Why fairness can’t be automated. Philosophy & Technology, 34(4), 1087–1107. https://doi.org/10.1007/s13347-021-00458-4
6. Information Commissioner’s Office. (2023). Guidance on AI and data protection. https://ico.org.uk/for-organisations/guide-to-data-protection/guide-to-the-general-data-protection-regulation-gdpr/artificial-intelligence/
7. European Commission. (2024). Proposal for a Regulation on Artificial Intelligence (AI Act). https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52021PC0206
8. International Organization for Standardization. (2023). ISO/IEC 23894:2023 Information technology — Artificial intelligence — Guidance on risk management. https://www.iso.org/standard/81234.html

## Metadata

- **Migration Status**: Ontology block enriched on 2025-11-12
- **Last Updated**: 2025-11-12
- **Review Status**: Comprehensive editorial review
- **Verification**: Academic sources verified
- **Regional Context**: UK/North England where applicable
