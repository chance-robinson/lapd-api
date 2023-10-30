from fastapi import FastAPI, HTTPException
import psycopg2
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

class PredictRequest(BaseModel):
    desc: list = []
    area: list = []
    range: int

class InputData(BaseModel):
    data: list
    cols: list

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

def predict_query_request(request):
    desc = request.desc   
    area = request.area
    number = request.range
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict')
def predict_data(data: InputData):
    try:
        cpm = CrimePredictionModel()
        query_req = cpm.runAll(data.data, data.cols, 1)
        return {"predict": query_req}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post('/predict2')
def predict_data(request: PredictRequest):
    try:
        query_req,cols = predict_query_request(request)
        cols_list = cols.split(', ')
        tot_crimes = execute_query(query_req)
        cpm = CrimePredictionModel()
        query_req = cpm.runAll(tot_crimes, cols_list, 1)
        return {"predict": query_req}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)