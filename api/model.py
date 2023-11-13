from pydantic import BaseModel

class WeeklyProjection(BaseModel):
    data_source: str
    player_id: int
    player_name: str
    player_position: str
    week: int
    standard_projected_points: float
    half_ppr_projected_points: float
    ppr_projected_points: float
