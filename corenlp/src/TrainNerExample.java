
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.sequences.SeqClassifierFlags;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;
/**
* This class is to do a named entity search against a trained NER model for CoreNLP analysis
*
* @author  Hong Zheng
* @version 1.0
* @since   2020-04-07
*/
public class TrainNerExample {
	  public static void main(String[] args) throws Exception{
		  // do a ner search against trained NER model and get its result
		  final List<Entity> entityList=getNerSearch("data/data1.txt","data/train-data.txt");
		  
		  System.out.println("Entities found as following:");
		  for (Entity entity:entityList) {
			  System.out.println(entity.getName()+"\t"+entity.getType());
		  }
	  }
	  
	  public static List<Entity> getNerSearch(String dataFilePaths,String trainDataFilePaths) throws Exception {
		  final ClassLoader loader =TrainNerExample.class.getClassLoader();
		  final URL trainDataFilePath=loader.getResource(trainDataFilePaths);
		  final String trainDataFile = trainDataFilePath.getPath().substring(1);//remove first useless letter
		  
		  //build a trained model
		  trainNER("train-ner-model.ser.gz","train-model.prop",trainDataFile);

	          // set up pipeline properties
                  final Properties props = new Properties();
	          props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner");

                  // add the trained model. It cannot disable other default models but overwrite others if same	    
	          props.setProperty("ner.model", "train-ner-model.ser.gz");

	          final URL dataFilePath=loader.getResource(dataFilePaths);
	          final String dataFile = dataFilePath.getPath().substring(1);//remove first useless letter
	          final String message=readFileAsString(dataFile);
	      
	          // set up pipeline
	          final StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	          // make an example document
	          final CoreDocument doc = new CoreDocument(message);

	          // annotate the document
	          pipeline.annotate(doc);
	      
		  Entity entity;
		  final List<Entity> entityList=new ArrayList<Entity>();

		  for (CoreEntityMention em : doc.entityMentions()) {
		    	entity=new Entity(em.text(),em.entityType());
		    	entityList.add(entity);
		  }
		    
		  return entityList;	  
	  }
  
  private static String readFileAsString(String fileName)throws Exception 
  { 
      return new String(Files.readAllBytes(Paths.get(fileName))); 
  } 
  
  private static void trainNER(String modelOutPath, String prop, String trainingFilepath) {
	   final Properties props = StringUtils.propFileToProperties(prop);
	   props.setProperty("serializeTo", modelOutPath);
	   
	   if (trainingFilepath != null) {
	       props.setProperty("trainFile", trainingFilepath);
	   }
	   final SeqClassifierFlags flags = new SeqClassifierFlags(props);
	   final CRFClassifier<CoreLabel> crf = new CRFClassifier<>(flags);
	   crf.train();
	   crf.serializeClassifier(modelOutPath);
	}
}
