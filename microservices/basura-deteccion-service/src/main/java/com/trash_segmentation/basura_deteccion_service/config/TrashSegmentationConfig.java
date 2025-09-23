package com.trash_segmentation.basura_deteccion_service.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import com.mongodb.ConnectionString;
import com.mongodb.MongoClientSettings;
import com.mongodb.ServerApi;
import com.mongodb.ServerApiVersion;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;

@Configuration
public class TrashSegmentationConfig {

    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder.build();
    }

    @Configuration
    public static class WebConfig implements WebMvcConfigurer {
        @SuppressWarnings("null")
        @Override
        public void addCorsMappings(CorsRegistry registry) {
            registry.addMapping("/api/**")
                    .allowedOrigins("http://localhost:5500")
                    .allowedMethods("GET","POST","PUT","DELETE");
        }
    }

    
    @Configuration
    public static class MongoConfig {

        @Value("${spring.data.mongodb.uri}")
        private String connectionUri;

        @Bean
        public MongoClient mongoClient() {
            ServerApi serverApi = ServerApi.builder()
                    .version(ServerApiVersion.V1)
                    .build();

            MongoClientSettings settings = MongoClientSettings.builder()
                    .applyConnectionString(new ConnectionString(connectionUri))
                    .serverApi(serverApi)
                    .build();

            return MongoClients.create(settings);
        }
    }

}
