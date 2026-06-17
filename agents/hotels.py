

"""
Hotels Agent

Specialized agent for searching and recommending accommodations.
Handles natural language queries about hotels, amenities, and locations.

- search_hotels
- get_hotel_recommendation
"""

# from pathlib import Path
# import sys

# if __package__ in {None, ""}:
#     sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain.tools import tool
from tools.mock_data import get_hotels


@tool
def search_hotels(
    destination: str,
    traveler_type: str | None = None,
    budget_per_night: float | None = None,
) -> str:
    
    """
    This tool is responsible for searching hotels based on the user's preferences.
    Args:
    - destination: The city or location where the user wants to find hotels (e.g., "Paris", "Tokyo").
    - traveler_type: The type of traveler (e.g., "family", "business", "solo") to tailor hotel recommendations.
    - budget_per_night: The maximum budget per night for the hotel (optional).
    Returns:
    - A string with hotel options that match the user's preferences.
    """

    hotels: list = get_hotels(destination)

    if not hotels:
        return f"No hotels found for {destination}. Please try a different destination or adjust your search parameters."
    
    # Filter hotels based on budget
    if budget_per_night:
        hotels = [hotel for hotel in hotels if hotel["price_per_night"] <= budget_per_night]

    # Filter hotels based on traveler type
    if traveler_type :
        hotels = [hotel for hotel in hotels if traveler_type.lower() in hotel.get("traveler_type", [])]

    

    results = [f"Found {len(hotels)} hotels in {destination} within a budget of {budget_per_night} for traveler type: {traveler_type}:\n"]

    for hotel in hotels:
        amenities_str = ", ".join(hotel["amenities"][:4])
        if len(hotel["amenities"]) > 4:
            amenities_str += f" +{len(hotel['amenities']) - 4} more"

        result = f"""
🏨 {hotel['name']}
   📍 Location: {hotel['neighborhood']}
   ⭐ Rating: {hotel['rating']}/5 ({hotel['reviews']} reviews)
   💰 Price: ${hotel['price_per_night']}/night
   🎯 Best for: {', '.join(hotel['traveler_type'])}
   ✨ Amenities: {amenities_str}
   📝 {hotel['description']}
"""
        results.append(result)

    return "\n".join(results)


@tool
def get_hotel_recommendation(
    destination: str,
    traveler_type: str,
    priority: str = "balanced"
) -> str:
    
    """Get a personalized hotel recommendation based on traveler profile.
    
    Args:
        destination: The destination city
        traveler_type: Type of traveler - "solo", "couples", "families", "luxury", "budget"
        priority: What to prioritize - "price", "rating", "location", or "balanced"
    
    Returns:
        Top hotel recommendation with explanation
    """

    hotels: list = get_hotels(destination)

    if not hotels:
        return f"No hotels found for {destination}. Please try a different destination or adjust your search parameters."
    
    # Filter hotels based on traveler type
    matching_hotels = [hotel for hotel in hotels if traveler_type.lower() in hotel.get("traveler_type", [])]
    if not matching_hotels:
        matching_hotels = hotels  # fallback to all hotels if no match
    
    #priority-based sorting
    if priority == "price":
        matching_hotels.sort(key=lambda x: x["price_per_night"])
    elif priority == "rating":
        matching_hotels.sort(key=lambda x: x["rating"], reverse=True)
    else:
        # balanced approach: sort by a combination of price and rating
        matching_hotels.sort(key=lambda x: (x["price_per_night"], -x["rating"]))

    top_pick = matching_hotels[0]

    result = f"""🌟 TOP RECOMMENDATION for {traveler_type} traveler(s) in {destination}:

    🏨 {top_pick['name']}
    📍 {top_pick['neighborhood']}
    ⭐ {top_pick['rating']}/5 ({top_pick['reviews']} reviews)
    💰 ${top_pick['price_per_night']}/night

    Why this hotel:
    • {top_pick['description']}
    • Amenities: {', '.join(top_pick['amenities'])}

    """
        # Add runner up if available
    if len(matching_hotels) > 1:
        runner_up = matching_hotels[1]
        result += f"""
    🥈 RUNNER-UP: {runner_up['name']}
    ${runner_up['price_per_night']}/night | ⭐ {runner_up['rating']}/5
    """
        
    return result


# if __name__ == "__main__":
#     print(get_hotel_recommendation(destination="paris", traveler_type="couples", priority="price"))


   
HOTELS_AGENT_PROMPT = """You are a hotel and accommodation specialist. Your job is to help users find the perfect place to stay.

Your capabilities:
- Search for available hotels in any destination
- Filter by budget, traveler type, and preferences
- Recommend hotels based on specific needs (families, couples, business, etc.)
- Provide insights on neighborhoods and locations

When responding:
1. Always consider the traveler type (solo, couples, families, etc.)
2. Factor in budget constraints when mentioned
3. Highlight what makes each hotel special
4. Consider location convenience for the type of trip
5. Mention key amenities relevant to the traveler's needs

Be helpful and specific. If someone is traveling with kids, prioritize family-friendly options. 
For couples, consider romantic or boutique hotels. For budget travelers, focus on value."""

def create_hotels_agent(model):
    """Create and return hotels agent"""
    return create_agent(
        model,
        tools=[search_hotels, get_hotel_recommendation],
        system_prompt=HOTELS_AGENT_PROMPT
    )