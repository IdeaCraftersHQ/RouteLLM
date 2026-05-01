# Test Index: tests

| Test File | What It Tests | Key Functions |
|-----------|---------------|----------------|
| `alloydb/alloydb_integration_test.go` | Alloydb | TestAlloyDBCreateCluster, TestAlloyDBCreateInstance, TestAlloyDBCreateUser (+ 1 more) |
| `alloydb/alloydb_wait_for_operation_test.go` | Alloydb Wait For Operation | TestWaitToolEndpoints |
| `alloydbainl/alloydb_ai_nl_integration_test.go` | Alloydb Ai Nl | TestAlloyDBAINLToolEndpoints |
| `alloydbomni/alloydb_omni_integration_test.go` | Alloydb Omni | TestAlloyDBOmni |
| `alloydbpg/alloydb_pg_integration_test.go` | Alloydb Pg | TestAlloyDBPgIAMConnection, TestAlloyDBPgIpConnection, TestAlloyDBPgToolEndpoints |
| `bigquery/bigquery_integration_test.go` | Bigquery | TestBigQueryToolEndpoints, TestBigQueryToolWithDatasetRestriction, TestBigQueryWriteModeAllowed (+ 2 more) |
| `bigtable/bigtable_integration_test.go` | Bigtable | TestBigtableToolEndpoints |
| `cassandra/cassandra_integration_test.go` | Cassandra | TestCassandra |
| `clickhouse/clickhouse_integration_test.go` | Clickhouse | TestClickHouse, TestClickHouseBasicConnection, TestClickHouseEdgeCases (+ 4 more) |
| `cloudgda/cloud_gda_integration_test.go` | Cloud Gda | TestCloudGdaToolEndpoints |
| `cloudhealthcare/cloud_healthcare_integration_test.go` | Cloud Healthcare | TestHealthcareToolEndpoints, TestHealthcareToolWithStoreRestriction |
| `cloudloggingadmin/cloud_logging_admin_integration_test.go` | Cloud Logging Admin | TestLogAdminToolEndpoints |
| `cloudmonitoring/cloud_monitoring_integration_test.go` | Cloud Monitoring | TestTool_Invoke, TestTool_Invoke_Error |
| `cloudsql/cloud_sql_clone_instance_test.go` | Cloud Sql Clone Instance | TestCloneInstanceToolEndpoints |
| `cloudsql/cloud_sql_create_backup_test.go` | Cloud Sql Create Backup | TestCreateBackupToolEndpoints |
| `cloudsql/cloud_sql_create_database_test.go` | Cloud Sql Create Database | TestCreateDatabaseToolEndpoints |
| `cloudsql/cloud_sql_create_users_test.go` | Cloud Sql Create Users | TestCreateUsersToolEndpoints |
| `cloudsql/cloud_sql_get_instances_test.go` | Cloud Sql Get Instances | TestGetInstancesToolEndpoints |
| `cloudsql/cloud_sql_list_databases_test.go` | Cloud Sql List Databases | TestListDatabasesToolEndpoints |
| `cloudsql/cloud_sql_restore_backup_test.go` | Cloud Sql Restore Backup | TestRestoreBackupToolEndpoints |
| `cloudsql/cloudsql_list_instances_test.go` | Cloudsql List Instances | TestListInstance |
| `cloudsql/cloudsql_wait_for_operation_test.go` | Cloudsql Wait For Operation | TestCloudSQLWaitToolEndpoints |
| `cloudsqlmssql/cloud_sql_mssql_create_instance_integration_test.go` | Cloud Sql Mssql Create Instance | TestCreateInstanceToolEndpoints |
| `cloudsqlmssql/cloud_sql_mssql_integration_test.go` | Cloud Sql Mssql | TestCloudSQLMSSQLIpConnection, TestCloudSQLMSSQLToolEndpoints |
| `cloudsqlmysql/cloud_sql_mysql_create_instance_integration_test.go` | Cloud Sql Mysql Create Instance | TestCreateInstanceToolEndpoints |
| `cloudsqlmysql/cloud_sql_mysql_integration_test.go` | Cloud Sql Mysql | TestCloudSQLMySQLIAMConnection, TestCloudSQLMySQLIpConnection, TestCloudSQLMySQLToolEndpoints |
| `cloudsqlpg/cloud_sql_pg_create_instances_test.go` | Cloud Sql Pg Create Instances | TestCreateInstanceToolEndpoints |
| `cloudsqlpg/cloud_sql_pg_integration_test.go` | Cloud Sql Pg | TestCloudSQLPgIAMConnection, TestCloudSQLPgIpConnection, TestCloudSQLPgSimpleToolEndpoints |
| `cloudsqlpg/cloud_sql_pg_upgrade_precheck_test.go` | Cloud Sql Pg Upgrade Precheck | TestPreCheckToolEndpoints |
| `cockroachdb/cockroachdb_integration_test.go` | Cockroachdb | TestCockroachDB |
| `couchbase/couchbase_integration_test.go` | Couchbase | TestCouchbaseToolEndpoints |
| `dataform/dataform_integration_test.go` | Dataform | TestDataformCompileTool |
| `dataplex/dataplex_integration_test.go` | Dataplex | TestDataplexToolEndpoints |
| `dataproc/dataproc_integration_test.go` | Dataproc | TestDataprocClustersToolEndpoints |
| `dgraph/dgraph_integration_test.go` | Dgraph | TestDgraphToolEndpoints |
| `elasticsearch/elasticsearch_integration_test.go` | Elasticsearch | TestElasticsearchToolEndpoints |
| `firebird/firebird_integration_test.go` | Firebird | TestFirebirdToolEndpoints |
| `firestore/firestore_integration_test.go` | Firestore | TestFirestoreToolEndpoints |
| `http/http_integration_test.go` | Http | TestHttpToolEndpoints |
| `looker/looker_integration_test.go` | Looker | TestLooker |
| `mariadb/mariadb_integration_test.go` | Mariadb | TestMySQLToolEndpoints |
| `mindsdb/mindsdb_integration_test.go` | Mindsdb | TestMindsDBToolEndpoints |
| `mongodb/mongodb_integration_test.go` | Mongodb | TestMongoDBToolEndpoints |
| `mssql/mssql_integration_test.go` | Mssql | TestMSSQLToolEndpoints |
| `mysql/mysql_integration_test.go` | Mysql | TestMySQLToolEndpoints |
| `neo4j/neo4j_integration_test.go` | Neo4J | TestNeo4jToolEndpoints |
| `oceanbase/oceanbase_integration_test.go` | Oceanbase | TestOceanBaseToolEndpoints |
| `oracle/oracle_integration_test.go` | Oracle | TestOracleSimpleToolEndpoints |
| `postgres/postgres_integration_test.go` | Postgres | TestPostgres |
| `prompts/custom/prompts_integration_test.go` | Prompts | TestMCPPromptsIntegration |
| `redis/redis_test.go` | Redis | TestRedisToolEndpoints |
| `serverlessspark/serverless_spark_integration_test.go` | Serverless Spark | TestServerlessSparkToolEndpoints |
| `singlestore/singlestore_integration_test.go` | Singlestore | TestSingleStoreToolEndpoints |
| `snowflake/snowflake_integration_test.go` | Snowflake | TestSnowflake |
| `spanner/spanner_integration_test.go` | Spanner | TestSpannerToolEndpoints |
| `sqlite/sqlite_integration_test.go` | Sqlite | TestSQLiteExecuteSqlTool, TestSQLiteToolEndpoint |
| `tidb/tidb_integration_test.go` | Tidb | TestTiDBToolEndpoints |
| `trino/trino_integration_test.go` | Trino | TestTrinoToolEndpoints |
| `valkey/valkey_test.go` | Valkey | TestValkeyToolEndpoints |
| `yugabytedb/yugabytedb_integration_test.go` | Yugabytedb | TestYugabyteDB |

⚠️ IMPORTANT: Keep this index up to date as tests are added/removed/modified. This document helps future maintainers understand test coverage at a glance.