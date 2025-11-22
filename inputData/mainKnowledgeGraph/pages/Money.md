- ### OntologyBlock

  - **Identification**
    - ontology:: true
    - is-subclass-of:: [[DisruptiveTechnology]]
    - term-id:: DT-0002
    - preferred-term:: Money
    - source-domain:: cross-domain
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-11-05

  - **Definition**
    - definition:: A medium of exchange, unit of account, and store of value accepted within an economic system, encompassing physical currency, digital money, and virtual currency forms.
    - maturity:: mature
    - source:: [[BIS]], [[IMF]], [[ISO 4217]], [[Bank of England]]
    - authority-score:: 1.0

  - **Semantic Classification**
    - owl:class:: dt:Money
    - owl:physicality:: HybridEntity
    - owl:role:: Object
    - owl:inferred-class:: dt:HybridObject
    - belongsToDomain:: [[EconomicSystemDomain]]
    - owl:disjointWith:: [[NeuralNetwork]], [[BlockchainDomain]], [[MetaverseDomain]]

  - #### OWL Restrictions
    - has-part some FiatCurrency
    - has-part some DigitalMoney
    - implemented-by some Stablecoin
    - has-part some CentralBankDigitalCurrency
    - requires some Trust
    - is-part-of some ValueTransfer
    - implemented-by some Cash
    - enables some PriceDiscovery
    - requires some AcceptanceNetwork
    - implemented-by some ElectronicMoney
    - enables some Savings
    - enables some EconomicExchange
    - has-part some Cryptocurrency
    - requires some IssuingAuthority
    - has-part some VirtualCurrency
    - implemented-by some BankDeposit
    - enables some Debt

  - #### CrossDomainBridges
    - bridges-to:: [[CentralBankDigitalCurrency]] via has-part
    - bridges-to:: [[DigitalMoney]] via has-part
    - bridges-to:: [[AcceptanceNetwork]] via requires
    - bridges-to:: [[Cryptocurrency]] via has-part
    - bridges-to:: [[VirtualCurrency]] via has-part

  - 