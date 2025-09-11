# Corporate Cyber-Threat Response Flowchart

```mermaid
graph TD
    subgraph "Phase 1: Initial Triage"
        A[🚨 Threat Detected: User reports<br> 'a funny-looking email'] --> B{Is it after 4:59 PM on a Friday?};
        B -- Yes --> C[Assign ticket severity: LOWEST<br>Add to next week's sprint];
        B -- No --> D{Does the user have a backup?};
    end

    subgraph "Phase 2: The Backup Inquisition"
        D -- Yes --> E[Instruct user to restore from backup];
        E --> F{Did the backup work?};
        F -- Yes --> G[Close ticket. Add note:<br>'User error resolved.'];
        F -- No --> H["Explain that backups should also be tested.<br>This is a teachable moment."];
        D -- No --> I["Deliver stern lecture on the critical importance of backups."];
        I --> J{Does the user seem remorseful?};
        J -- No --> K[Hang up abruptly];
        J -- Yes --> L[Sigh audibly. Create new ticket for 'Backup Policy Training'.];
    end

    subgraph "Phase 3: Escalation & Analysis Paralysis"
        C --> M[Monday Morning: Re-evaluate ticket];
        G --> N((End));
        H --> M;
        L --> M;
        M --> O{Is the system *actually* on fire?};
        O -- No --> P[Run automated vulnerability scan.<br>Schedule 3-hour meeting to discuss results.];
        O -- Yes --> Q{Have we blamed the user yet?};
        Q -- No --> I;
        Q -- Yes --> R[Engage external high-cost consultants];
    end

    subgraph "Phase 4: Resolution Theatre"
        P --> S[Meeting concludes: Action item is to schedule another meeting.];
        R --> T[Consultants produce a 200-page PDF report.<br>Cost: $150,000];
        T --> U["File report in 'Compliance' Sharepoint folder.<br>Mark as unread."];
        S --> V((End));
        U --> W[Follow up with an email to the<br>Cyber Threat Prevention Helpline.];
        W --> X((End));
        K --> Y((End));
    end

    style A fill:#ff4d4d,stroke:#333,stroke-width:2px
    style C fill:#fffacd,stroke:#333,stroke-width:2px
    style G fill:#d4edda,stroke:#333,stroke-width:2px
    style K fill:#f8d7da,stroke:#333,stroke-width:2px
    style R fill:#e2d9f3,stroke:#333,stroke-width:2px
    style T fill:#e2d9f3,stroke:#333,stroke-width:2px
    style N fill:#d4edda,stroke:#333,stroke-width:4px
    style V fill:#d4edda,stroke:#333,stroke-width:4px
    style X fill:#d4edda,stroke:#333,stroke-width:4px
    style Y fill:#d4edda,stroke:#333,stroke-width:4px