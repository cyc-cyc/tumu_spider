
import mysql.connector

def delete_from_information_extraction(record_id, host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        cursor = conn.cursor()

        # 定义删除 SQL 语句
        delete_query = "DELETE FROM information_extraction WHERE id = %s"
        
        # 执行删除操作
        cursor.execute(delete_query, (record_id,))
        conn.commit()  # 提交事务

        if cursor.rowcount > 0:
            print("成功删除数据。")
        else:
            print("未找到符合条件的记录。")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # 关闭连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()
def fetch_data_from_mysql(host, user, password, database):
    # 连接到 MySQL 数据库
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = conn.cursor()

    # 执行查询
    cursor.execute("SELECT * FROM information_extraction")  # 替换为你的表名
    rows = cursor.fetchall()

    # 打印结果
    for row in rows:
        print(row)

    # 关闭连接
    cursor.close()
    conn.close()

def insert_into_information_extraction(data, host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        cursor = conn.cursor()

        # 定义插入 SQL 语句
        insert_query = """
        INSERT INTO information_extraction (
            id,
            building_name,
            detailed_address,
            neighborhood,
            street,
            building_function,
            structure_type,
            longitude,
            latitude,
            above_ground_floors,
            underground_floors,
            age,
            completion_year,
            compute_year,
            features,
            surrounding_transportation,
            surrounding_facilities,
            total_households,
            households_per_floor,
            square_meter_planning,
            layout_planning,
            management_mode,
            building_morphology
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        # 执行插入操作
        cursor.execute(insert_query, data)
        conn.commit()  # 提交事务

        print("成功插入数据。")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # 关闭连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def fetch_column_names_from_table(table_name,host, user, password, database):
    try:
        # 连接到 MySQL 数据库
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        cursor = conn.cursor()

        # 执行查询以获取列名
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = cursor.fetchall()

        # 打印结果
        print(f"表 '{table_name}' 的列名:")
        for column in columns:
            print(column[0])  # 每个列信息是一个元组，取第一个元素为列名

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        # 关闭连接
        if cursor:
            cursor.close()
        if conn:
            conn.close()
# fetch_data_from_mysql('b101.guhk.cc', 'tumu', 'TJtumu', 'tumu')
fetch_column_names_from_table("information_extraction",'b101.guhk.cc', 'tumu', 'TJtumu', 'PearAdminFlask')



# 使用示例
data_to_insert = (
    29,  # id
    'Building A',  # building_name
    '123 Main St',  # detailed_address
    'Downtown',  # neighborhood
    'Main St',  # street
    'Residential',  # building_function
    'Concrete',  # structure_type
    120.0,  # longitude
    30.0,  # latitude
    10,  # above_ground_floors
    2,  # underground_floors
    5,  # age
    2000,  # completion_year
    2023,  # compute_year
    'Newly renovated',  # features
    'Subway, Bus',  # surrounding_transportation
    'Park, Grocery Store',  # surrounding_facilities
    50,  # total_households
    5,  # households_per_floor
    100,  # square_meter_planning
    '3BHK',  # layout_planning
    'Managed',  # management_mode
    'Modern'  # building_morphology
)
print(len(data_to_insert))

# data_to_insert=( 71,'存在冲突：新北市板桥区莒光国民小学 / 力麒御品',  '莒光路334', '板桥区', '莒光路', '国民小学', '未提及', '未提及', '未提及', '未提及', '未提及', '截至2024年, 46年(1978年-2024年)', '1978年', '', '', '', '', '', '', '', '', '', '')
delete_from_information_extraction(1000, 'b101.guhk.cc', 'tumu', 'TJtumu', 'PearAdminFlask')
# print(len(data_to_insert))

# insert_into_information_extraction(data_to_insert,'b101.guhk.cc', 'tumu', 'TJtumu', 'PearAdminFlask')
input()
fetch_data_from_mysql('b101.guhk.cc', 'tumu', 'TJtumu', 'PearAdminFlask')
