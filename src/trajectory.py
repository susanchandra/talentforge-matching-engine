"""
src/trajectory.py — Career seniority timeline analysis for TalentForge.

Analyzes the arc of a candidate's career progression, computing
growth velocity and classifying trajectory type.
Also provides a Plotly timeline chart visualization.
"""

from typing import List

import plotly.graph_objects as go


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SENIORITY_SCALE = {
    "intern": 1,
    "junior": 2,
    "mid": 3,
    "senior": 4,
    "staff": 5,
    "principal": 6,
}

SENIORITY_LABELS = {v: k.capitalize() for k, v in SENIORITY_SCALE.items()}

# Required seniority level → numeric scale value
JD_SENIORITY_MAP = {
    "junior": 2,
    "mid": 3,
    "senior": 4,
    "staff": 5,
    "principal": 6,
}


class TrajectoryAnalyzer:
    """
    Analyzes a candidate's career progression from their role history.

    Computes growth velocity (seniority delta per year), classifies
    trajectory type, and generates a normalized trajectory_score.
    """

    def analyze(self, role_history: List[dict]) -> dict:
        """
        Analyze a candidate's career trajectory from their role history.

        Args:
            role_history: List of role dicts, each with keys:
                          title (str), company (str),
                          duration_years (float), seniority_level (str).
                          Input order is treated as chronological order.

        Returns:
            Dict with keys:
            - initial_level (int): Seniority scale value of first role
            - final_level (int): Seniority scale value of last role
            - total_years (float): Sum of all role durations
            - delta_per_year (float): Seniority growth rate per year
            - trajectory_class (str): Classification of growth pattern
            - trajectory_score (float): Normalized 0-1 score
            - role_sequence (list): Simplified role list for charting
        """
        if not role_history:
            return {
                "initial_level": 2,
                "final_level": 2,
                "total_years": 0.0,
                "delta_per_year": 0.0,
                "trajectory_class": "unknown",
                "trajectory_score": 0.3,
                "role_sequence": [],
            }

        # Map each role to its seniority scale value (default to 'mid'=3 if unknown)
        role_sequence = []
        for role in role_history:
            level_str = role.get("seniority_level", "mid").lower()
            level_num = SENIORITY_SCALE.get(level_str, 3)
            duration = float(role.get("duration_years", 1.0) or 1.0)
            role_sequence.append({
                "title": role.get("title", "Unknown Role"),
                "seniority_level": level_str,
                "seniority_num": level_num,
                "duration_years": duration,
            })

        initial_level = role_sequence[0]["seniority_num"]
        final_level = role_sequence[-1]["seniority_num"]
        total_years = sum(r["duration_years"] for r in role_sequence)
        seniority_delta = final_level - initial_level

        # Compute growth rate (handle zero-duration edge case)
        delta_per_year = seniority_delta / total_years if total_years > 0 else 0.0

        # Classify trajectory type
        trajectory_class = self._classify_trajectory(delta_per_year)

        # Compute normalized trajectory score
        trajectory_score = self._compute_trajectory_score(delta_per_year)

        return {
            "initial_level": initial_level,
            "final_level": final_level,
            "total_years": round(total_years, 2),
            "delta_per_year": round(delta_per_year, 4),
            "trajectory_class": trajectory_class,
            "trajectory_score": round(trajectory_score, 4),
            "role_sequence": role_sequence,
        }

    def _classify_trajectory(self, delta_per_year: float) -> str:
        """Classify career trajectory based on seniority growth rate."""
        if delta_per_year >= 0.5:
            return "high_growth"
        elif delta_per_year >= 0.1:
            return "steady_growth"
        elif delta_per_year > -0.1:
            return "lateral"
        else:
            return "declining"

    def _compute_trajectory_score(self, delta_per_year: float) -> float:
        """
        Normalize delta_per_year to a 0-1 trajectory score.

        Mapping:
        - delta_per_year <= 0: score = 0.0 (declining or flat)
        - delta_per_year = 0: score = 0.3 (lateral, baseline)
        - delta_per_year >= 1.0: score = 1.0 (exceptional growth)
        - Linear interpolation between 0 and 1 for values in between.
        """
        if delta_per_year <= 0:
            return max(0.0, 0.3 + delta_per_year * 0.3)  # declining caps at 0
        elif delta_per_year >= 1.0:
            return 1.0
        else:
            # Linear scale: 0 maps to 0.3, 1.0 maps to 1.0
            return 0.3 + delta_per_year * 0.7

    def build_timeline_chart(
        self,
        trajectory_data: dict,
        candidate_name: str,
        jd_seniority: str,
    ) -> go.Figure:
        """
        Build a Plotly line chart showing the candidate's career progression.

        Args:
            trajectory_data: Output of self.analyze().
            candidate_name: Candidate's name for the chart title.
            jd_seniority: JD's required seniority level string.

        Returns:
            Plotly Figure object.
        """
        role_sequence = trajectory_data.get("role_sequence", [])

        if not role_sequence:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No role history available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
            return fig

        # Build cumulative timeline x-axis
        x_years = [0.0]
        y_levels = [role_sequence[0]["seniority_num"]]
        cumulative = 0.0

        for role in role_sequence:
            cumulative += role["duration_years"]
            x_years.append(cumulative)
            y_levels.append(role["seniority_num"])

        # Get role labels for hover
        role_labels = ["Career Start"] + [r["title"] for r in role_sequence]

        # JD reference line level
        jd_level = JD_SENIORITY_MAP.get(jd_seniority.lower(), 4)

        # Build figure
        fig = go.Figure()

        # Career progression line
        fig.add_trace(go.Scatter(
            x=x_years,
            y=y_levels,
            mode="lines+markers",
            name=candidate_name,
            line=dict(color="#1a1a2e", width=3),
            marker=dict(size=10, color="#4a90d9", symbol="circle"),
            text=role_labels,
            hovertemplate="<b>%{text}</b><br>Year: %{x:.1f}<br>Level: %{customdata}<extra></extra>",
            customdata=[SENIORITY_LABELS.get(y, str(y)) for y in y_levels],
        ))

        # JD requirement reference line
        fig.add_hline(
            y=jd_level,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"JD Required: {jd_seniority.capitalize()}",
            annotation_position="right",
            annotation_font_color="#e74c3c",
        )

        # Styling
        fig.update_layout(
            title=dict(
                text=f"{candidate_name} — Career Trajectory",
                font=dict(size=16, color="#1a1a2e"),
            ),
            xaxis=dict(
                title="Years of Experience",
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title="Seniority Level",
                tickvals=list(SENIORITY_LABELS.keys()),
                ticktext=list(SENIORITY_LABELS.values()),
                showgrid=True,
                gridcolor="#f0f0f0",
                range=[0.5, 6.5],
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            height=320,
            margin=dict(l=80, r=120, t=50, b=50),
        )

        return fig
