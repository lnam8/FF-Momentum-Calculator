from pydantic import BaseModel
from typing import Optional, Union

class WeeklyProjection(BaseModel):
    data_source: str
    player_id: int
    player_name: str
    player_position: str
    week: int
    standard_projected_points: float
    half_ppr_projected_points: float
    ppr_projected_points: float

class WeeklyMadden(BaseModel):
    year: Optional[str]
    week: Optional[str]
    college: Optional[str]
    signingBonus_diff: Union[int, str]
    awareness_rating: Union[int, str]
    shortRouteRunning_diff: Union[int, str]
    press_diff: Union[int, str]
    carrying_diff: Union[int, str]
    strength_rating: Union[int, str]
    catchInTraffic_rating: Union[int, str]
    pursuit_rating: Union[int, str]
    plyrAssetname: Optional[str]
    breakSack_diff: Union[int, str]
    plyrPortrait_diff: Union[int, str]
    catching_rating: Union[int, str]
    spinMove_rating: Union[int, str]
    acceleration_diff: Union[int, str]
    breakTackle_diff: Union[int, str]
    height: Union[int, str]
    finesseMoves_rating: Union[int, str]
    strength_diff: Union[int, str]
    runBlock_rating: Union[int, str]
    tackle_rating: Union[int, str]
    runBlock_diff: Union[int, str]
    kickPower_diff: Union[int, str]
    zoneCoverage_rating: Union[int, str]
    plyrBirthdate: Optional[str]
    awareness_diff: Union[int, str]
    runningStyle_rating: Optional[str]
    totalSalary: Union[int, str]
    trucking_rating: Union[int, str]
    toughness_diff: Union[int, str]
    hitPower_diff: Union[int, str]
    tackle_diff: Union[int, str]
    jukeMove_rating: Union[int, str]
    playRecognition_rating: Union[int, str]
    shortRouteRunning_rating: Union[int, str]
    status: Optional[str]
    lastName: Optional[str]
    jerseyNum_diff: Union[int, str]
    jerseyNum: Union[int, str]
    breakSack_rating: Union[int, str]
    passBlockFinesse_diff: Union[int, str]
    jumping_rating: Union[int, str]
    throwAccuracyDeep_diff: Union[int, str]
    stamina_diff: Union[int, str]
    throwAccuracyShort_diff: Union[int, str]
    powerMoves_diff: Union[int, str]
    throwOnTheRun_diff: Union[int, str]
    zoneCoverage_diff: Union[int, str]
    jukeMove_diff: Union[int, str]
    speed_diff: Union[int, str]
    release_rating: Union[int, str]
    agility_diff: Union[int, str]
    hitPower_rating: Union[int, str]
    throwAccuracyMid_rating: Union[int, str]
    kickAccuracy_rating: Union[int, str]
    impactBlocking_diff: Union[int, str]
    stamina_rating: Union[int, str]
    plyrPortrait: Optional[str]
    kickPower_rating: Union[int, str]
    throwUnderPressure_rating: Union[int, str]
    team: Optional[str]
    signingBonus: Union[int, str]
    height_diff: Union[int, str]
    playAction_diff: Union[int, str]
    throwUnderPressure_diff: Union[int, str]
    changeOfDirection_diff: Union[int, str]
    blockShedding_rating: Union[int, str]
    fullNameForSearch: Optional[str]
    overall_rating: Union[int, str]
    deepRouteRunning_diff: Union[int, str]
    passBlockFinesse_rating: Union[int, str]
    runBlockFinesse_diff: Union[int, str]
    throwPower_rating: Union[int, str]
    kickReturn_rating: Union[int, str]
    leadBlock_rating: Union[int, str]
    bCVision_rating: Union[int, str]
    primaryKey_diff: Union[int, str]
    mediumRouteRunning_diff: Union[int, str]
    playAction_rating: Union[int, str]
    totalSalary_diff: Union[int, str]
    teamId_diff: Union[int, str]
    leadBlock_diff: Union[int, str] 
    catchInTraffic_diff: Union[int, str]
    mediumRouteRunning_rating: Union[int, str]
    acceleration_rating: Union[int, str]
    spinMove_diff: Union[int, str]
    yearsPro_diff: Union[int, str]
    spectacularCatch_rating: Union[int, str]
    injury_rating: Union[int, str]
    weight: Union[int, str]
    playRecognition_diff: Union[int, str]
    deepRouteRunning_rating: Union[int, str]
    firstName: Optional[str]
    yearsPro: Union[int, str]
    manCoverage_diff: Union[int, str]
    catching_diff: Union[int, str]
    throwAccuracyShort_rating: Union[int, str]
    position: Optional[str]
    overall_diff: Union[int, str]
    weight_diff: Union[int, str]
    bCVision_diff: Union[int, str]
    throwPower_diff: Union[int, str]
    speed_rating: Union[int, str]
    runBlockPower_rating: Union[int, str]
    injury_diff: Union[int, str]
    toughness_rating: Union[int, str]
    throwOnTheRun_rating: Union[int, str]
    jumping_diff: Union[int, str]
    spectacularCatch_diff: Union[int, str]
    manCoverage_rating: Union[int, str]
    stiffArm_rating: Union[int, str]
    throwAccuracyMid_diff: Union[int, str]
    trucking_diff: Union[int, str]
    passBlock_diff: Union[int, str]
    powerMoves_rating: Union[int, str]
    iteration: Optional[str]
    stiffArm_diff: Union[int, str]
    passBlockPower_rating: Union[int, str]
    impactBlocking_rating: Union[int, str]
    carrying_rating: Union[int, str]
    breakTackle_rating: Union[int, str]
    plyrHandedness: Optional[str]
    kickReturn_diff: Union[int, str]
    passBlock_rating: Union[int, str]
    changeOfDirection_rating: Union[int, str]
    press_rating: Union[int, str]
    passBlockPower_diff: Union[int, str]
    pursuit_diff: Union[int, str]
    release_diff: Union[int, str]
    throwAccuracyDeep_rating: Union[int, str]
    age_diff: Union[int, str]
    archetype: Optional[str]
    runBlockPower_diff: Union[int, str]
    runBlockFinesse_rating: Union[int, str]
    finesseMoves_diff: Union[int, str]
    blockShedding_diff: Union[int, str]
    kickAccuracy_diff: Union[int, str]
    teamId: Union[int, str]
    agility_rating: Union[int, str]
    age: Union[int, str]
    primaryKey: Union[int, str]