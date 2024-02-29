import openai
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

openai.api_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


def get_response(company_name):
    def find_product_section(company_name):
        driver = webdriver.Firefox()  # 使用Chrome浏览器，确保已安装Chrome驱动程序并将其路径添加到系统PATH中
        driver.get("https://www.bing.com")
        search_box = driver.find_element("name", "q")
        search_box.send_keys(company_name + "官网")
        search_box.send_keys(Keys.RETURN)

        search_box.submit()
        driver.implicitly_wait(10)  # 等待10秒钟
        try:
            official_website = driver.find_element(By.CSS_SELECTOR, "li.b_algo h2 a").get_attribute("href")
            driver.get(official_website)
            
            # 等待一段时间，确保页面加载完成
            driver.implicitly_wait(5)
            driver.get(official_website)
    
            try:
                product_link = driver.find_element("link text", "Products")
            except:
                product_link = None 
    
            if product_link == None:
                try:
                    product_link = driver.find_element("link text", "产品中心")
                except:
                    product_link = None
    
            if product_link == None:
                try:
                    product_link = driver.find_element("link text", "产品")
                except:
                    product_link = None

            if product_link == None:
                try:
                    product_link = driver.find_element("link text", "产品系列")
                except:
                    product_link = None
            if product_link == None:
                try:
                    product_link = driver.find_element("link text", "产品展示")
                except:
                    product_link = None
            if product_link == None:
                try:
                    product_link = driver.find_element("link text", "产品世界")
                except:
                    product_link = None
            if product_link == None:
                try:
                    product_link = driver.find_element("link text", "产品与服务")
                except:
                    product_link = None
            if product_link == None:
                try:
                    product_link = driver.find_element("link text", "产品与应用")
                except:
                    product_link = None
            product_url = product_link.get_attribute("href")
    
    
        except Exception as e:
            print("Error:", e)
            product_url = None
        
        driver.quit()
        return product_url
    # 示例用法
    print("Company name: ", company_name)
    print("\n")
    product_section_link = find_product_section(company_name)
    if product_section_link:
        print(f"The product section link of {company_name} is: {product_section_link}")
    else:
        print(f"Product section link not found for {company_name}")
    
    driver = webdriver.Chrome()
    driver.get(product_section_link)
    #
    # 获取页面文本内容并打印
    page_text = driver.find_element("tag name", 'body').text
    print("页面文本内容：", page_text)
    
    question_system_prompt = """你是一个产品专家，下面是一个页面介绍，麻烦介绍一下页面中的产品"""
    prompt = "请根据下面的网页页面文本内容介绍其对应的产品：\n" + page_text + "\n 只需要给出产品名字，不需要给出其它信息。"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": question_system_prompt},
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )
    answer = response["choices"][0]["message"]["content"]
    return answer

import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('modified_data.xlsx')
print(df)
# 遍历非标题的每一行
for index, row in df.iterrows():
    if pd.isnull(row[1]):
        try:
            answer = get_response(row[0])
        except:
            answer = None
        print("answer: ", answer)
        df.at[index, "主要产品"] = answer

df.to_excel('modified_data2.xlsx', index=False)
