package com.click.example;

import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.options.Validation.Required;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.testing.TestPipeline;
import com.google.api.services.bigquery.model.TableRow;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.io.jdbc.JdbcIO;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import org.apache.commons.dbcp2.BasicDataSource;

public class BQuerySqlFlow {      // This class should be public and the filename and class name should be same starting with capital letter


  public interface SqlJobOptions extends PipelineOptions {     // This class is used when we pass the options from the command line using -- 

  }

  static class SqlOutputSetter implements JdbcIO.PreparedStatementSetter<TableRow>   // This class is used to assign values to '?' in insert statement
  {
    private static final long serialVersionUID = 1L;

    public void setParameters(TableRow element, PreparedStatement query) throws Exception  // We receive the data from BigQuery in TableRow 
    {

      String name = (String) element.get("Name");
      String sex = (String) element.get("Sex");

      query.setString(1, name);  // Sets 1st ? to name
      query.setString(2, sex);	 // Sets 1st ? to name

    }
  }

  public static void main(String[] args) {

      SqlJobOptions options = PipelineOptionsFactory.fromArgs(args).withValidation().as(SqlJobOptions.class);
	  
      Pipeline p = Pipeline.create(options);   // Creating a pipeline

	  // ProjectId: dark-pipe-247510; BQDataset = Titanic; BQTable = titanic
      p
      .apply(BigQueryIO.read().from("dark-pipe-247510:Titanic.titanic"))   // Reads data from BigQuery
      .apply(JdbcIO.<TableRow>write()									   // Writes data to SQL
              .withDataSourceConfiguration(JdbcIO.DataSourceConfiguration.create("com.mysql.jdbc.Driver", "jdbc:mysql://google/Customer?cloudSqlInstance=dark-pipe-247510:us-central1:database&socketFactory=com.google.cloud.sql.mysql.SocketFactory&user=root&password=root&useSSL=false")
                      )   // SPecifies details for Sql destination
              .withStatement("insert into titanic values(?,?)")  // We are inserting only two columns from the BigQuery data to Sql 
              .withPreparedStatementSetter(new SqlOutputSetter())); 
    p.run().waitUntilFinish();
  }

}