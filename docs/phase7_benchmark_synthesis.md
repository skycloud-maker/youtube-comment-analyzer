# PHASE 7 Benchmark Synthesis (Strategy-First IA)

This note captures only information architecture patterns used for `dashboard_v2`.
It does not copy visual design from any external product.

## Source patterns and how they were applied

1. Overview first, details on demand  
Source: [Power BI dashboard design tips](https://learn.microsoft.com/en-us/power-bi/create-reports/service-dashboards-design-tips)  
Pattern: show key decision objects first, place highest-level information in the first scan area, keep details for drill-down.  
Applied in v2: landing tab opens on strategic issues and priority signals; evidence is moved to Layer 3 drill-down.

2. Decision object must be explicit and prioritized  
Source: [Atlassian incident severity levels](https://www.atlassian.com/incident-management/kpis/severity-levels), [JSM impact/urgency matrix](https://support.atlassian.com/jira-service-management-cloud/docs/how-impact-and-urgency-are-used-to-calculate-priority/)  
Pattern: users need a clear severity/priority object before reading raw logs.  
Applied in v2: issue rows include `priority_signal`, `issue_state`, `supporting_cluster_count`, and `impact_signal` before any comment text.

3. Evidence hierarchy (signal -> grouped evidence -> source)  
Source: [Productboard insights boards](https://support.productboard.com/hc/en-us/articles/360056354634-Use-insights-boards-to-group-related-notes), [Productboard feedback-to-insight linking](https://support.productboard.com/hc/en-us/articles/360056354514-Link-user-feedback-to-related-feature-ideas-using-insights)  
Pattern: consolidate evidence into grouped insight objects, then connect back to underlying notes.  
Applied in v2: strategy issue is primary object; representative/similar/raw comments are attached in evidence drawers.

4. Keep primary canvas uncluttered; reduce source-level dominance  
Source: [Tableau dashboard best practices](https://help.tableau.com/current/pro/desktop/en-us/dashboards_best_practices.htm), [Tableau effective dashboards blog](https://www.tableau.com/blog/best-practices-for-building-effective-dashboards)  
Pattern: avoid overcrowding, limit front-layer views, move supporting details lower in the flow.  
Applied in v2: representative cards are no longer landing object; raw comments are hidden in expanded drill-down sections.

5. Journey-stage framing for strategic interpretation  
Source: [Qualtrics customer journey stages guide](https://www.qualtrics.com/articles/customer-experience/customer-journey-stages/)  
Pattern: interpret feedback by journey stage, not only by isolated sentiment/topic labels.  
Applied in v2: each issue is shown with journey concentration and judgment-axis concentration as first-class fields.

