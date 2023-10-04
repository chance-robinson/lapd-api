from fastapi import FastAPI, HTTPException
import matplotlib.pyplot as plt
from io import BytesIO
import psycopg2
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib.pyplot as plt
import base64
import matplotlib
from ml2 import CrimePredictionModel

matplotlib.use('agg')

DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'lapd'
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins if required
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Request(BaseModel):
    desc: list = []
    area: list = []
    number: str
    startDate: str
    endDate: str

def execute_query(query):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


def query_request(request):
    desc = request.desc   
    area = request.area
    startDate = request.startDate
    endDate = request.endDate
    number = request.number
    area_conditions = []
    area_col = []
    for idx, item in enumerate(area):
        value = item['value'][0]
        condition = f"OR area_name = '{value}'"
        if idx == 0:
            condition = f"WHERE (area_name = '{value}'"
        elif idx == len(area) - 1:
            condition = f"{condition} OR area_name = '{value}')"
        if len(area) == 1:
            condition = f"WHERE area_name = '{value}'"
        area_conditions.append(condition)
        area_col.append(value)
    # Creating variables for 'desc'
    desc_conditions = []
    desc_col = []
    for idx, item in enumerate(desc):
        value = item['value'][0]
        condition = f"OR crm_cd_desc = '{value}'"
        if idx == 0:
            condition = f"AND (crm_cd_desc = '{value}'"
        if idx == len(desc) - 1:
            condition += ")"
        if len(desc) == 1:
            condition = f"AND crm_cd_desc = '{value}'"
        desc_conditions.append(condition)
        desc_col.append(value)

    area_condition_str = ' '.join(area_conditions)
    desc_condition_str = ' '.join(desc_conditions)
    cols = 'crm_cd_desc, area_name, date_occ, location, lat, long, dr_no'
    if startDate == 'na':
        query = f"""
        SELECT {cols}
        FROM public.crime
        JOIN public."crimeCode" ON public.crime.crm_cd = public."crimeCode".crm_cd
        JOIN public.area ON public.crime.area = public.area.area
        {area_condition_str}
            {desc_condition_str}
            AND date_occ >= current_date - interval '{number} days'
        ORDER BY date_occ DESC;
        """
    else:
        query = f"""
        SELECT {cols}
        FROM public.crime
        JOIN public."crimeCode" ON public.crime.crm_cd = public."crimeCode".crm_cd
        JOIN public.area ON public.crime.area = public.area.area
        {area_condition_str}
            {desc_condition_str}
            AND date_occ >= DATE '{startDate}' AND date_occ < DATE '{endDate}'
        ORDER BY date_occ DESC;
        """
    return query,cols

@app.get("/")
def root():
    # Replace the placeholder query with your actual query
    query = 'SELECT crm_cd_desc FROM public."crimeCode"'
    desc = execute_query(query)
    query = "SELECT area_name FROM public.area"
    area = execute_query(query)
    return {"desc": desc, "area": area}

@app.post('/data')
def plot_data(request: Request):
    try:
        query_req,cols = query_request(request)
        tot_crimes = execute_query(query_req)
        return {"crimes": tot_crimes, "cols": cols}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
class InputData(BaseModel):
    data: list
    cols: list

@app.post('/predict')
def predict_data(data: InputData):
    try:
        cpm = CrimePredictionModel()
        query_req = cpm.runAll(data.data, data.cols, 1)
        return {"predict": query_req}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))



@app.post('/plot')
def plot_data(request: Request):
    try:
        query_req,cols = query_request(request)
        tot_crimes = execute_query(query_req)

        dates = list(set([entry[2].date().isoformat() for entry in tot_crimes]))  # Extract the date and convert to ISO format
        areas = list(set([entry[1] for entry in tot_crimes]))  # Assuming area_name is the second element (index 1)
        crimes = list(set([entry[0] for entry in tot_crimes]))  # Assuming area_name is the second element (index 1)
        # Count the occurrences of each area on each date
        counts = {date: {area: {crime: 0 for crime in crimes} for area in areas} for date in dates}

        for entry in tot_crimes:
            date = entry[2].date().isoformat()  # Extract the date and convert to ISO format
            area = entry[1]  # Assuming area_name is the second element (index 1)
            crime = entry[0]  # Assuming crime description is the first element (index 0)
            counts[date][area][crime] += 1
        # Prepare the stacked bar chart data

        dates = list(counts.keys())
        areas = list(counts[dates[0]].keys())
        crimes = list(counts[dates[0]][areas[0]].keys())
        dates.sort()
        # Create a dictionary to store the color mapping for each crime
        crime_colors = {crime: plt.cm.get_cmap('tab10')(idx) for idx, crime in enumerate(crimes)}


        num_bars = len(dates)
        bar_width = 0.35

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)

        index = 0
        for i, date in enumerate(dates):
            crime_counts = [sum(counts[date][area][crime] for area in areas) for crime in crimes]
            bottom = [sum(crime_counts[:k]) for k in range(len(crime_counts))]
            for idx, val in enumerate(crime_counts):
                ax.bar(
                    index, val, bar_width, bottom=bottom[idx],
                    label=crimes[idx], color=crime_colors[crimes[idx]]
                )
            index += 1

        ax.set_xlabel('Date')
        ax.set_ylabel('Counts')
        ax.set_title(f'Combined Crime Counts in {areas} by Date')
        ax.set_xticks(range(num_bars))
        ax.set_xticklabels(dates)
        plt.xticks(rotation = 30) # Rotates X-Axis Ticks by 45-degrees
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = list(set(labels))
        unique_handles = [handles[labels.index(label)] for label in unique_labels]
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(unique_handles, unique_labels, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode()

        return {"crimes": tot_crimes, "plot": encoded_image, "cols": cols}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)