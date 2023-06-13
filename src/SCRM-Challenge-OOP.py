#!/usr/bin/env python
# coding: utf-8

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

class DataProcessor:
    def __init__(self, spark):
        self.spark = spark

    def read_data(self, format, path, options=None):
        """
        Lee los datos de un archivo en el formato especificado y ruta dada.

        Args:
            format (str): Formato de los datos (ej. "json", "csv").
            path (str): Ruta del archivo de datos.
            options (dict, optional): Opciones adicionales para la lectura de datos. Default es None.

        Returns:
            pyspark.sql.DataFrame: DataFrame que contiene los datos leídos.
        """
        if options is None:
            options = {}
        return self.spark.read.format(format).options(**options).load(path)

    def calculate_total_quantity(self, data_df, group_by_column, quantity_column):
        """
        Calcula la cantidad total agrupada por una columna en un DataFrame.

        Args:
            data_df (pyspark.sql.DataFrame): DataFrame de entrada.
            group_by_column (str): Nombre de la columna por la cual agrupar.
            quantity_column (str): Nombre de la columna de cantidad.

        Returns:
            pyspark.sql.DataFrame: DataFrame con la cantidad total por grupo.
        """
        return data_df.groupBy(group_by_column).agg(sum(quantity_column).alias("total_quantity"))

    def join_tables(self, left_df, right_df, join_columns, select_columns):
        """
        Realiza una operación de join entre dos DataFrames.

        Args:
            left_df (pyspark.sql.DataFrame): DataFrame izquierdo.
            right_df (pyspark.sql.DataFrame): DataFrame derecho.
            join_columns (list): Lista de columnas para realizar el join.
            select_columns (list): Lista de columnas para seleccionar en el DataFrame resultante.

        Returns:
            pyspark.sql.DataFrame: DataFrame resultante después del join y selección de columnas.
        """
        return left_df.join(right_df, join_columns).select(select_columns)

    def count_distinct_values(self, data_df, group_by_columns, count_column, alias_name):
        """
        Calcula el número de valores distintos en una columna en un DataFrame.

        Args:
            data_df (pyspark.sql.DataFrame): DataFrame de entrada.
            group_by_columns (list): Lista de columnas por las cuales agrupar.
            count_column (str): Nombre de la columna para contar valores distintos.
            alias_name (str): Nombre del alias para el resultado.

        Returns:
            pyspark.sql.DataFrame: DataFrame con el conteo de valores distintos por grupo.
        """
        return data_df.groupBy(group_by_columns).agg(countDistinct(count_column).alias(alias_name))

    def calculate_ranked_values(self, data_df, group_by_columns, value_column, rank_column):
        """
        Calcula los valores clasificados por grupos en un DataFrame.

        Args:
            data_df (pyspark.sql.DataFrame): DataFrame de entrada.
            group_by_columns (list): Lista de columnas por las cuales agrupar.
            value_column (str): Nombre de la columna de valores.
            rank_column (str): Nombre de la columna para el rango.

        Returns:
            pyspark.sql.DataFrame: DataFrame con los valores clasificados por grupos y rango.
        """
        window_spec = Window.partitionBy(group_by_columns).orderBy(desc(value_column))
        return data_df.groupBy(group_by_columns).agg(sum(value_column).alias(value_column)) \
                       .withColumn(rank_column, row_number().over(window_spec))

    def filter_by_rank(self, data_df, rank_column, rank):
        """
        Filtra un DataFrame por un rango específico.

        Args:
            data_df (pyspark.sql.DataFrame): DataFrame de entrada.
            rank_column (str): Nombre de la columna de rango.
            rank (int): Valor del rango a filtrar.

        Returns:
            pyspark.sql.DataFrame: DataFrame filtrado por rango.
        """
        return data_df.filter(col(rank_column) == rank)

    def group_values(self, data_df, group_by_columns, aggregate_column, aggregate_function, alias_name):
        """
        Agrupa los valores en un DataFrame y aplica una función de agregación.

        Args:
            data_df (pyspark.sql.DataFrame): DataFrame de entrada.
            group_by_columns (list): Lista de columnas por las cuales agrupar.
            aggregate_column (str): Nombre de la columna a agregar.
            aggregate_function (pyspark.sql.functions): Función de agregación a aplicar.
            alias_name (str): Nombre del alias para el resultado.

        Returns:
            pyspark.sql.DataFrame: DataFrame con los valores agrupados y agregados.
        """
        return data_df.groupBy(group_by_columns).agg(aggregate_function(aggregate_column).alias(alias_name))

    def union_dataframes(self, df1, df2):
        """
        Une dos DataFrames en uno.

        Args:
            df1 (pyspark.sql.DataFrame): Primer DataFrame.
            df2 (pyspark.sql.DataFrame): Segundo DataFrame.

        Returns:
            pyspark.sql.DataFrame: DataFrame resultante de la unión de los dos DataFrames.
        """
        return df1.union(df2)

    def process_data(self, data_paths, formats, options=None):
        """
        Procesa los datos utilizando las operaciones definidas en la clase.

        Args:
            data_paths (list): Lista de rutas de los archivos de datos.
            formats (list): Lista de formatos de los archivos de datos.
            options (list, optional): Lista de opciones adicionales para la lectura de datos. Default es None.

        Returns:
            tuple: Tupla de DataFrames resultantes después del procesamiento.
        """
        dfs = []
        for path, format in zip(data_paths, formats):
            df = self.read_data(format, path, options)
            dfs.append(df)

        joined_df = self.join_tables(dfs[1], dfs[0], ["product_id"], ["ticket_id", "product_name", "country"])
        total_quantity_df = self.calculate_total_quantity(dfs[1], "product_id", "quantity")
        distinct_stores_df = self.count_distinct_values(dfs[1].join(dfs[0], ["product_id", "store_id"]),
                                                        "product_id", "store_id", "num_stores")
        ranked_stores_df = self.calculate_ranked_values(dfs[1].join(dfs[0], ["store_id"]),
                                                        ["product_id", "store_id"], "quantity", "rank")
        second_most_selling_df = self.filter_by_rank(ranked_stores_df, "rank", 2)
        grouped_stores_df = self.group_values(second_most_selling_df.join(dfs[0], ["product_id"]),
                                              "category_name", "store_id", collect_list, "stores")
        unioned_stores_df = self.union_dataframes(dfs[0], dfs[2])

        return joined_df, total_quantity_df, distinct_stores_df, grouped_stores_df, unioned_stores_df


def main():
    spark = SparkSession.builder.getOrCreate()
    data_processor = DataProcessor(spark)
    data_paths = [r"C:\Users\lgoicoch\Documents\Proyectos Luis\SCRM\data\products.json",
                  r"C:\Users\lgoicoch\Documents\Proyectos Luis\SCRM\data\ticket_line.csv",
                  r"C:\Users\lgoicoch\Documents\Proyectos Luis\SCRM\data\stores.csv",
                  r"C:\Users\lgoicoch\Documents\Proyectos Luis\SCRM\data\stores_v2.csv"]
    formats = ["json", "csv", "csv", "csv"]
    options = [{"header": "true"}, {"header": "true"}, {"header": "true"}, {"header": "true"}]
    joined_df, total_quantity_df, distinct_stores_df, grouped_stores_df, unioned_stores_df = \
        data_processor.process_data(data_paths, formats, options)
    # Do something with the processed data


if __name__ == "__main__":
    main()