"""
Itinerary Agent

Specialized agent for organizing travel components into a cohesive day-by-day schedule.
Handles optimization of timing, logistics, and flow.

- create_daily_schedule
- optimize_route
- generate_trip_summary
"""

from langchain.agents import create_agent
from langchain.tools import tool

@tool
def create_daily_schedule(
    activities: str,
    hotel_location: str,
    trip_pace: str = "moderate"
) -> str:
    """Create an optimized daily schedule from a list of activities.
    
    Args:
        activities: Comma-separated list of activities/places to visit
        hotel_location: The neighborhood where the hotel is located
        trip_pace: "relaxed", "moderate", or "packed"
    
    Returns:
        Organized daily schedule with timing suggestions
    """

    activity_list = [a.strip() for a in activities.split(",")]

    # Defines time slots based on pace
    if trip_pace == "relaxed":
        slots_per_day = 2
        time_slots = ["10:00 AM - 1:00 PM", "3:00 PM - 6:00 PM"]
    elif trip_pace == "packed":
        slots_per_day = 4
        time_slots = ["8:00 AM - 10:30 AM", "11:00 AM - 1:30 PM", "2:30 PM - 5:00 PM", "6:00 PM - 9:00 PM"]
    else:  # moderate
        slots_per_day = 3
        time_slots = ["9:00 AM - 12:00 PM", "2:00 PM - 5:00 PM", "7:00 PM - 9:00 PM"]

    # Create Schedule
    result = f"""📅 DAILY SCHEDULE (Pace: {trip_pace})
🏨 Starting from: {hotel_location}

"""
    
    day = 1
    slot_index = 0

    for activity in activity_list:
        if slot_index == 0: 
            result += f"-- DAY {day} -- \n"

        result += f"⏱️ {time_slots[slot_index]}: {activity}\n"

        slot_index += 1
        if slot_index >= len(time_slots):
            result += "\n"
            slot_index = 0
            day += 1

    return result

@tool
def optimize_route(locations: str) -> str:
    """Suggest an optimized order for visiting multiple locations.
    
    Args:
        locations: Comma-separated list of neighborhoods/areas to visit
    
    Returns:
        Suggested visiting order with reasoning
    """

    location_list = [loc.strip() for loc in locations.split(",")]

    # Demo results (hardcoded-in for demo purposes - Tokyo based)
    # Use proper location analysis tool in real-world scenario
    result = """🗺️ OPTIMIZED ROUTE SUGGESTION:

"""
    
    # For demo purposes, just provide logical grouping advice
    result += f"📍 Locations to visit: {', '.join(location_list)}\n\n"
    result += "💡 Optimization tips:\n"
    result += "• Group nearby locations together to minimize travel time\n"
    result += "• Visit eastern areas (Asakusa, Ueno) in the morning\n"
    result += "• Central areas (Ginza, Tsukiji) work well mid-day\n"
    result += "• Western areas (Shinjuku, Shibuya) are great for evening/nightlife\n"
    result += "• Save Odaiba/Disney area for a dedicated day trip\n\n"
    
    result += f"Suggested order: {' → '.join(location_list)}\n"

    return result

@tool
def generate_trip_summary(
    destination: str,
    num_days: int,
    flight_info: str,
    hotel_info: str,
    activities_info: str,
    total_budget: int | None = None
) -> str:
    """Generate a comprehensive trip summary with all components.
    
    Args:
        destination: The destination city
        num_days: Number of days for the trip
        flight_info: Summary of selected flight
        hotel_info: Summary of selected hotel
        activities_info: Summary of planned activities
        total_budget: Total trip budget in USD (optional)
    
    Returns:
        Complete trip summary document
    """

    result = f"""
═══════════════════════════════════════════════════════════════
                    🌏 TRIP TO {destination.upper()} 🌏
                         {num_days}-Day Itinerary
═══════════════════════════════════════════════════════════════

✈️ FLIGHT DETAILS
───────────────────────────────────────────────────────────────
{flight_info}

🏨 ACCOMMODATION
───────────────────────────────────────────────────────────────
{hotel_info}

🎯 ACTIVITIES & EXPERIENCES
───────────────────────────────────────────────────────────────
{activities_info}

"""
    
    if total_budget:
        result += f"""
💰 BUDGET OVERVIEW
───────────────────────────────────────────────────────────────
Total Budget: ${total_budget:,} USD
(Detailed breakdown available upon request)

"""
    
    result += """
📝 TRAVEL TIPS
───────────────────────────────────────────────────────────────
• Check passport validity (6+ months recommended)
• Consider travel insurance
• Download offline maps
• Learn a few basic local phrases
• Keep copies of important documents

═══════════════════════════════════════════════════════════════
                    Have an amazing trip! 🎉
═══════════════════════════════════════════════════════════════
"""

    return result


ITINERARY_AGENT_PROMPT = """You are an expert travel itinerary planner. Your job is to organize all travel components into a logical, enjoyable schedule.

Your capabilities:
- Create day-by-day schedules from a list of activities
- Optimize routes to minimize travel time
- Generate comprehensive trip summaries
- Balance activities for an enjoyable pace

When creating itineraries:
1. Consider logical geographic flow (don't zigzag across the city)
2. Account for travel time between locations
3. Build in rest time and meals
4. Put high-energy activities earlier in the day
5. Group nearby attractions together
6. Consider opening hours and best times to visit

Your goal is to create a realistic, enjoyable schedule - not an exhausting checklist. 
Quality experiences matter more than quantity."""


def create_itinerary_agent(model):
    """Create and return the itinerary agent"""

    return create_agent(
        model,
        tools=[create_daily_schedule, optimize_route, generate_trip_summary],
        system_prompt=ITINERARY_AGENT_PROMPT
    )

