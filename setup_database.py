import os
import pymysql

from dotenv import load_dotenv


load_dotenv()


MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "localhost"
)

MYSQL_PORT = int(
    os.getenv(
        "MYSQL_PORT",
        "3306"
    )
)

MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "root"
)

MYSQL_PASSWORD = os.getenv(
    "MYSQL_PASSWORD",
    ""
)

MYSQL_DATABASE = os.getenv(
    "MYSQL_DATABASE",
    "peakdemand"
)


print("=" * 60)
print("PeakDemand Database Setup")
print("=" * 60)


try:

    print("\nConnecting to MySQL...")

    connection = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )

    cursor = connection.cursor()

    print("MySQL connection successful.")

    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`"
    )

    connection.commit()

    print(
        f"Database '{MYSQL_DATABASE}' created successfully."
    )

    cursor.close()
    connection.close()

    print("\nDatabase setup completed.")
    print(
        "Now run: python run.py"
    )

except Exception as error:

    print("\nDATABASE ERROR")
    print(error)

    print(
        "\nCheck your .env MySQL settings."
    )